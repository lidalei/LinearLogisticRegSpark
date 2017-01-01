package classification

import org.apache.spark.ml.feature.{PolynomialExpansion, VectorAssembler}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.sql.types.{DoubleType, StructField, StructType}
import org.apache.spark.ml.linalg.{DenseMatrix, Matrices, Vector, Vectors}
import org.apache.spark.rdd.RDD
import Helper.VectorMatrixManipulation._
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.util.LongAccumulator
import Helper.InstanceUtilities._
import hyperparameter.tuning.MyCrossValidation.{kFoldRDD, kFoldDF}
import org.apache.spark.ml.param.{DoubleParam, IntParam, ParamMap, ParamValidators}
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.storage.StorageLevel
import scala.annotation.tailrec

/**
  * Created by Sophie on 11/23/16.
  */


object MyLogisticRegression {

  /**
    * Sigmoid function
    * @param z Independent variable
    * @return Dependent variable
    */
  private def sigmoid(z: Double): Double  = {
    // http://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    if(z < 0) {
      val expZ = math.exp(z)
      expZ / (1.0 + expZ)
    }
    else {
      1.0 / (1.0 + math.exp(-z))
    }
  }

  /**
    * Compute the gradient of total cross entropy in terms of parameters, Vector theta.
    * @param data Used to compute gradient
    * @param theta in the point of theta
    * @param intercept and intercept
    * @return Pair (gradient of theta, gradient of intercept)
    */
  private def computeGradient(data: RDD[Instance], theta: Vector, intercept: Double): (Vector, Double) = {
    // compute gradient
    data.map{
      case Instance(label, features) => {
        val gradientFactor: Double = sigmoid(vecInnerProduct(theta, features) + intercept) - label
        // theta and intercept
        (vecScale(features, gradientFactor), gradientFactor)
      }
    }.treeReduce((pair1: (Vector, Double), pair2: (Vector, Double)) => (vecAdd(pair1._1, pair2._1), pair1._2 + pair2._2))
  }

  /** From package org.apache.spark.mllib.util.MLUtils
    * When `x` is positive and large, computing `math.log(1 + math.exp(x))` will lead to arithmetic
    * overflow. This will happen when `x > 709.78` which is not a very large number.
    * It can be addressed by rewriting the formula into `x + math.log1p(math.exp(-x))` when `x > 0`.
    * @param x a floating-point value as input.
    * @return the result of `math.log(1 + math.exp(x))`.
    */
  private def log1pExp(x: Double): Double = {
    if (x > 0) {
      x + math.log1p(math.exp(-x))
    } else {
      math.log1p(math.exp(x))
    }
  }

  /**
    * Compute mean cross entropy instead of total cross entropy such that the result is comparable with Spark ML
    * @param data Used to compute cross entropy
    * @param theta in the point of theta
    * @param intercept and intercept
    * @return cross entropy
    */
  private def crossEntropy(data: RDD[Instance], theta: Vector, intercept: Double): Double = {
    // numerical stability, be cautious with zero and one, using log sum trick
    // https://lingpipe-blog.com/2012/02/16/howprevent-overflow-underflow-logistic-regression/
    data.map{
      case Instance(label, features) => {
        val z: Double = vecInnerProduct(theta, features) + intercept

        if(label == 1.0) log1pExp(-z) else log1pExp(z)
      }
    }.mean
  }


  /**
    * Compute the gradient of total cross entropy in terms of parameters, Vector theta.
    * And total cross entropy over the passed theta and intercept. And number of instances.
    * @param data Used to compute gradient
    * @param theta in the point of theta
    * @param intercept and intercept
    * @return Pair (gradient of theta, gradient of intercept, total cross entropy, number of instances)
    */
  private def computeGradientAndCrossEntropy(data: RDD[Instance], theta: Vector, intercept: Double): (Vector, Double, Double, Long) = {
    // compute gradient
    data.map{
      case Instance(label, features) => {
        val z: Double = vecInnerProduct(theta, features) + intercept

        val xEntropy = if(label == 1.0) log1pExp(-z) else log1pExp(z)

        val gradientFactor: Double = sigmoid(z) - label
        // theta and intercept
        (vecScale(features, gradientFactor), gradientFactor, xEntropy, 1L)
      }
    }.treeReduce((pair1, pair2) => (vecAdd(pair1._1, pair2._1), pair1._2 + pair2._2, pair1._3 + pair2._3, pair1._4 + pair2._4))
  }


  /**
    * Compute Hessian matrix
    * @param data Data
    * @param theta contains intercept as the last element
    * @return Hessian matrix, X.T * S * X
    */
  private def computeHessianMatrix(data: RDD[Instance], theta: Vector): Vector = {
    data.map({
      case Instance(label: Double, features: Vector) => {
        val sigma = sigmoid(vecInnerProduct(theta, features))
        vecScale(outerVecProduct(features), sigma * (1 - sigma))
      }
    }).treeReduce(vecAdd)
  }


  /**
    * MyLogisticRegression train, Gradient descent or Newton's method
    * Somewhere the bias should be considered...
    * @param trainingMethod "Gradient" or "Newton"
    * @param trainData If gradient descent, intercept is fit separately. If Newton's, features contains all-one column
    * @param batchSize The number of batches, default 1, only applies on Gradient descent now
    * @param maxIterations The maximum number of iterations
    * @param learningRate The gradient descent (Newton's method) step
    * @param lambda l2 regularization
    * @param tolerance Iteration tolerance
    * @return fit coefficients, theta plus intercept plus training losts
    */
  def train(trainingMethod: String = "Gradient", batchSize: Int = 1)
           (trainData: RDD[Instance], maxIterations: Int, learningRate: Double, lambda: Double, tolerance: Double = 1e-6): (Vector, Double, Array[Double]) = {

    require(trainingMethod == "Gradient" || trainingMethod == "Newton", "Only support Gradient descent and Newton's method.")
    require(batchSize >= 1 && maxIterations >= 1 && lambda >= 0.0 && tolerance >= 0.0)

    val handlePersistence = trainData.getStorageLevel == StorageLevel.NONE
    if(handlePersistence) {
      trainData.persist(StorageLevel.MEMORY_AND_DISK)
    }

    val numberOfFeatures: Int = trainData.first match {
      case Instance(_, features) => features.size
    }

    // set initial coefficients and intercept
    var theta: Vector = Vectors.zeros(numberOfFeatures)
    var intercept: Double = 0.0

    val lost = Array.fill[Double](maxIterations)(0.0)
    var decayedLearningRate = learningRate

    var (i, deltaLost): (Int, Double) = (0, 100.0)

    // TODO implement mini-batch or combine computeGradient and crossEntropy together
    if(trainingMethod == "Gradient") {
//      if(batchSize > 1) {
//        // divide data into batchSize parts
//
//      }
      while(i < maxIterations && deltaLost > tolerance) {
        if(i >= 2 && lost(i - 1) > lost(i - 2)) {
          decayedLearningRate = decayedLearningRate / 2.0
        }

        // gradient of cross entropy at theta and intercept and cross entropy
        val (gradientOfThetaXE, gradientOfInterceptXE, crossEntropy, num) = computeGradientAndCrossEntropy(trainData, theta, intercept)
        lost(i) = crossEntropy / num

        // Add l2 regularization. Do not regularize intercept
        val gradientOfThetaRegularizer: Vector = vecScale(theta, lambda)

        // descent
        val deltaTheta = vecScale(vecAdd(gradientOfThetaXE, gradientOfThetaRegularizer), -decayedLearningRate)
        theta = vecAdd(theta, deltaTheta)

        val deltaIntercept = -decayedLearningRate * gradientOfInterceptXE
        intercept = intercept + deltaIntercept

        // change of delta or change of cross entropy
        if(i >= 1) {
          deltaLost = math.abs(lost(i) - lost(i - 1))
        }

        i += 1
      }
    }
    else {
      // keep intercept as zero
      intercept = 0.0

      while(i < maxIterations && deltaLost > tolerance) {
        if(i >= 2 && lost(i - 1) > lost(i - 2)) {
          decayedLearningRate = decayedLearningRate / 2.0
        }
        // gradient of cross entropy at theta and intercept and cross entropy
        val (gradientOfTheta, _, crossEntropy, num) = computeGradientAndCrossEntropy(trainData, theta, intercept)
        lost(i) = crossEntropy / num

        // compute Hessian matrix
        val hessianMatrixVec = computeHessianMatrix(trainData, theta)

        val eyeMatrixArr: Array[Double] = DenseMatrix.eye(numberOfFeatures).toArray

        // size = numberOfFeatures * numberOfFeatures
        val invHessianMatrixArr: Array[Double] = solveLinearEquations(hessianMatrixVec.toArray, eyeMatrixArr, numberOfFeatures, numberOfFeatures)

        val deltaNewton = Matrices.dense(numberOfFeatures, numberOfFeatures, invHessianMatrixArr).multiply(gradientOfTheta)

        val deltaTheta = vecScale(deltaNewton, -decayedLearningRate)

        theta = vecAdd(theta, deltaTheta)

        if(i >= 1) {
          // change of delta or change of cross entropy
          deltaLost = lost(i) - lost(i - 1)
        }

        i += 1
      }
    }

    if(handlePersistence) {
      trainData.unpersist()
    }

    // remove un-computed lost if early stop
    (theta, intercept, lost.slice(0, i))
  }


  /**
    * Make predictions and compute classification metrics
    * @param testData The data to make predictions on
    * @param threshold The prediction threshold, default 0.5
    * @return Predictions and confusion matrix
    */
  def predict(testData: RDD[Instance], theta: Vector, intercept: Double, threshold: Double = 0.5): (RDD[InstanceWithPrediction], ConfusionMatrix) = {
    require(threshold > 0 && threshold < 1)

    // make predictions, lazy operation, be cautious, accumulator here might be updated more times due to failure of tasks
    val predictions: RDD[InstanceWithPrediction] = testData.map{
      case Instance(label, features) => {
        val predictionProb = sigmoid(vecInnerProduct(theta, features) + intercept)
        val prediction: Double = if(predictionProb >= threshold) 1.0 else 0.0

        InstanceWithPrediction(label, predictionProb, prediction)
      }
    }

    // compute confusion matrix, using ACTION to ensure the correctness
    val sc: SparkContext = SparkContext.getOrCreate()

    // accumulators to store confusion matrix
    val truePositiveAcc: LongAccumulator = sc.longAccumulator("truePositive")
    val falsePositiveAcc: LongAccumulator = sc.longAccumulator("falsePositive")
    val trueNegativeAcc: LongAccumulator = sc.longAccumulator("trueNegative")
    val falseNegativeAcc: LongAccumulator = sc.longAccumulator("falseNegative")

    // when foreach is executed on a parallelism collection, here rdd, it will be executed in parallel
    predictions.foreach{
      case InstanceWithPrediction(label, _, prediction) => {
        if(prediction == 1.0 && label == 1.0) {
          truePositiveAcc.add(1)
        }
        else if(prediction == 0.0 && label == 0.0) {
          trueNegativeAcc.add(1)
        }
        else if(prediction == 0.0 && label == 1.0) {
          falseNegativeAcc.add(1)
        }
        else {
          falsePositiveAcc.add(1)
        }
      }
    }

    // only driver program can get the value
    // since an action is executed, truePositiveAcc.value will get return the correct value
    (predictions, ConfusionMatrix(truePositiveAcc.value, trueNegativeAcc.value, falsePositiveAcc.value, falseNegativeAcc.value))
  }


  /**
    * Append polynomial expansion columns
    * @param df The data frame to be appended
    * @param polyDegrees The list of polynomial degrees to be appended
    * @param prefix The prefix of appended column names
    * @return Appended data frame
    */
  @tailrec
  def appendPolyExpand(df: DataFrame, polyDegrees: List[Int], featuresVecName: String, prefix: String): DataFrame = {
    if(polyDegrees.isEmpty) {
      df
    }
    else {
      // polynomial expansion
      val polyDegree = polyDegrees.head
      val polyNomialExpansion: PolynomialExpansion = new PolynomialExpansion().setInputCol(featuresVecName)
        .setOutputCol(prefix + polyDegree).setDegree(polyDegree)

      appendPolyExpand(polyNomialExpansion.transform(df), polyDegrees.tail, featuresVecName, prefix)
    }
  }


  /**
    * Cross validation to do hyperparameter tuning for logistic regression
    * @param trainingMethod "Gradient" or "Newton"
    * @param batchSize Int, the number of batches, at least one, and default 1, which represents full gradient descent
    * @param dataDF DataFrame used to optimize hyperparameters and train a "best" model usign the optimal hyperparameters
    * @param featuresVecName The column (features vector) used to make predictions
    * @param label The target column name, default "label"
    * @param k The number of folds used in cross validation, default 3-Fold
    * @param paramGrid The search space of hyperparameters
    * @param lambdaDoubleParam lambda, used in l2 regularization
    * @param polyDegreeIntParam polynomial expansion degree
    * @param scoreType The metric deciding the optimal hyperparameters
    * @param maxIterations Maximum number of iterations
    * @param learningRate Gradient descent (Newton's method) step
    * @param threshold Prediction threshold, default is 0.5
    * @return Best theta, intercept, training lost using, this best hyperparameters, (parameters, mean cross validation score)
    */
  def crossValidation(trainingMethod: String = "Gradient", batchSize: Int = 1)
                     (dataDF: DataFrame, featuresVecName: String, label: String = "label", k: Int = 3)
                     (paramGrid: Array[ParamMap], lambdaDoubleParam: DoubleParam, polyDegreeIntParam: IntParam, scoreType: String)
                     (maxIterations: Int, learningRate: Double, threshold: Double = 0.5): (Vector, Double, Array[Double], ParamMap, Array[(ParamMap, Double)]) = {

    var cvScores = Array[(ParamMap, Double)]()

    var polyDegreesSet: Set[Int] = Set[Int]()
    var lambdasSet: Set[Double] = Set[Double]()

    paramGrid.foreach((paramMap: ParamMap) => {
      val lambda: Double = paramMap.getOrElse[Double](lambdaDoubleParam, 0.0)
      val polyDegree: Int = paramMap.getOrElse[Int](polyDegreeIntParam, 1)

      polyDegreesSet += polyDegree
      lambdasSet += lambda
    })

    // excluding polynomial expansion with degree one
    polyDegreesSet -= 1

    val polyDegrees: List[Int] = polyDegreesSet.toList
    val lambdas: List[Double] = lambdasSet.toList

    // append all needed polynoial expansion columns
    val featuresPolyDegreeNamePrefix: String = "featuresVecPolyDegree"
    val polyExpandedDF: DataFrame = appendPolyExpand(dataDF, polyDegrees, featuresVecName, featuresPolyDegreeNamePrefix)

    polyExpandedDF.cache()

    val kFoldsDF: Array[(DataFrame, DataFrame)] = kFoldDF(polyExpandedDF, k)

    // search over the hyper-parameter grid spanned by l2 penalty and polynomial expansion
    // outer loop - grid search
    if(false) { // naive grid search
      // if there is no data transformation, use this
      cvScores = paramGrid.map((paramMap: ParamMap) => {
        val lambda: Double = paramMap.getOrElse[Double](lambdaDoubleParam, 0.0)
        val polyDegree: Int = paramMap.getOrElse[Int](polyDegreeIntParam, 1)

        val featuresName: String = if(polyDegree != 1) featuresPolyDegreeNamePrefix + polyDegree else featuresVecName

        // inner loop - cross validation
        val meanScore: Double = kFoldsDF.map{
          case (trainDataDF: DataFrame, testDataDF: DataFrame) => {
            val trainData = df2RDD(trainDataDF, featuresName, label, trainingMethod != "Gradient")
            val testData = df2RDD(testDataDF, featuresName, label, trainingMethod != "Gradient")

            // add lambda
            val (theta: Vector, intercept: Double, lost: Array[Double]) = train(trainingMethod)(trainData, maxIterations, learningRate, lambda)

            val (predictions: RDD[InstanceWithPrediction], confusionMatrix: ConfusionMatrix) = predict(testData, theta, intercept, threshold)
            score(confusionMatrix, scoreType)
          }}.sum / k

        (paramMap, meanScore)
      })
    }
    else {
      var _cvScores = Array[((Int, Double), Double)]()

      // outer loop - loop through polynomial expansion degree
      // sacrifice parallelism for less memory consumption
      for (polyDegree <- polyDegrees)
      {

        val featuresName: String = if(polyDegree != 1) featuresPolyDegreeNamePrefix + polyDegree else featuresVecName

        // inner loop - cross validation
        for {
          fold <- kFoldsDF
        }
        {
          val (trainDataDF, testDataDF) = fold
          val trainData: RDD[Instance] = df2RDD(trainDataDF, featuresName, label, trainingMethod != "Gradient")
          val testData: RDD[Instance] = df2RDD(testDataDF, featuresName, label, trainingMethod != "Gradient")

          trainData.persist()
          testData.persist()

          // ONLY parallelize lambda
          for {
            lambda <- lambdas
          }
          {
            val (theta: Vector, intercept: Double, _) = train(trainingMethod)(trainData, maxIterations, learningRate, lambda)
            val (_, confusionMatrix: ConfusionMatrix) = predict(testData, theta, intercept, threshold)

            _cvScores ++= Array(((polyDegree, lambda), score(confusionMatrix, scoreType)))
          }

//          val polyDegreeLambdasScores: Array[((Int, Double), Double)] = lambdas.map((lambda: Double) => {
//            val (theta: Vector, intercept: Double, _) = train(trainingMethod)(trainData, maxIterations, learningRate, lambda)
//            val (_, confusionMatrix: ConfusionMatrix) = predict(testData, theta, intercept, threshold)
//
//            ((polyDegree, lambda), score(confusionMatrix, scoreType))
//          }).toArray

//          _cvScores ++= polyDegreeLambdasScores
        }
      }

      cvScores = _cvScores.groupBy(_._1).mapValues(_.map(_._2).sum / k)
        .map((cvScore: ((Int, Double), Double))=> {
          (new ParamMap().put[Int](polyDegreeIntParam, cvScore._1._1).put[Double](lambdaDoubleParam, cvScore._1._2), cvScore._2)
        }).toArray
    }


    val bestParaMap: ParamMap = cvScores.maxBy[Double](_._2)._1
    val bestLambda: Double = bestParaMap.getOrElse[Double](lambdaDoubleParam, 0.0)
    val bestPolyDegree: Int = bestParaMap.getOrElse[Int](polyDegreeIntParam, 1)

    val bestFeaturesName = if(bestPolyDegree != 1) featuresPolyDegreeNamePrefix + bestPolyDegree else featuresVecName

    // train a model in the best feature
    val dataRDD: RDD[Instance] = df2RDD(polyExpandedDF, bestFeaturesName, label, false)

    val (theta: Vector, intercept: Double, lost: Array[Double]) = train(trainingMethod)(dataRDD, maxIterations, learningRate, bestLambda)

    (theta, intercept, lost, bestParaMap, cvScores)
  }


  /**
    * Starting point of the program
    * @param args, args(0) is the path to the dataset
    */
  def main(args: Array[String]): Unit = {
    // Do not use all the cores. TODO, comment setMaster to run in a cluster
    val conf = new SparkConf()
      .setAppName("My Logistic Regression").set("spark.executor.cores", "3") // no larger than 5 cores
      //.setMaster("local[2]")

    val sc = initializeSC(conf)
    val sparkSql = initializeSparkSession(conf)

    sc.setLogLevel("WARN")

    import sparkSql.implicits._

    // whether to display the result of mllib logistic regression
    val displayMLLogReg: Boolean = false
    val displayMyLogReg: Boolean = false
    val displayCV: Boolean = true

    // Gradient or Newton
    val trainingMethod: String = "Gradient"

    // low, high or all. The features used to make predictions, TODO, only use low level features now.
    val independentFeatures: String = "high"

    // check the contents of dataset
    var filePath = "/Users/Sophie/Downloads/MPML-Datasets/HIGGS_sample.csv"

    // the same file should be in all nodes of a cluster, including the master and workers
    if(args.length >= 1) {
      filePath = args(0)
    }

    // read data
    val cols: Array[String] = Array("label") ++ (1 to 28).toArray.map("feature" + _)
    val fields: Array[StructField] = cols.map(StructField(_, DoubleType, nullable = true))
    val schema = StructType(fields)

    val dataDF = sparkSql.read.option(key = "header", value = false).schema(schema).csv(filePath)

    /* begin data preparation */

    val featuresVecName: String = "featuresVec"
    var featuresCols = Array[String]()

    if(independentFeatures == "low") {
      // form a vector of all low level features
      featuresCols = cols.slice(1, 22)
    }
    else if(independentFeatures == "high") {
      // form a vector of all high level features
      featuresCols = cols.slice(22, 29)
    }
    else {
      // form a vector of all level features
      featuresCols = cols
    }

    val featuresVecAssembler = new VectorAssembler().setInputCols(featuresCols).setOutputCol(featuresVecName)

    // split data into train and test, TODO, change train / test
    val trainTestSplitArr: Array[DataFrame] = dataDF.randomSplit(Array(0.7, 0.3))
    val (trainDataDF, testDataDF) = (trainTestSplitArr(0), trainTestSplitArr(1))

    // transform train and test data and drop useless columns (col1 to col28)
    val trTrainDataDF = featuresVecAssembler.transform(trainDataDF).drop(cols.tail: _*)

    val trTestDataDF = featuresVecAssembler.transform(testDataDF).drop(cols.tail: _*)

    /* end data preparation */

    /* begin train */

    if(displayMLLogReg) {
      // mllib logistic regression, very similar with Newton's method

      val trainingStartTime = System.nanoTime()

      val logisticRegression: LogisticRegression = new LogisticRegression().setLabelCol("label")
        .setFeaturesCol(featuresVecName).setMaxIter(100).setFitIntercept(true)

      val mlLogisticRegModel: LogisticRegressionModel = logisticRegression.fit(trTrainDataDF)

      val trainingDuration = (System.nanoTime() - trainingStartTime) / 1e9d
      println("ML training duration: " + trainingDuration + " s.")

      val mlTheta: Vector = mlLogisticRegModel.coefficients
      val mlIntercept: Double = mlLogisticRegModel.intercept
      println("ML Theta: " + mlTheta.toArray.mkString(", "))
      println("ML Intercept: " + mlIntercept)
      println("ML Objective history: " + mlLogisticRegModel.summary.objectiveHistory.mkString(", "))

      val mlPredictions = mlLogisticRegModel.transform(trTestDataDF)
      println("ML Predictions: ")
      mlPredictions.select(mlLogisticRegModel.getProbabilityCol, mlLogisticRegModel.getPredictionCol).take(20).foreach(println)

      val biMetrics = new MulticlassMetrics(mlPredictions.select(mlLogisticRegModel.getPredictionCol, "label").rdd.map{
        case Row(prediction: Double, label: Double) => (prediction, label)
      })

      println("ML accuracy: " + biMetrics.accuracy)
    }

    if(displayMyLogReg) {
      // My Logistic Regression implementations
      // if the last parameter is true, add all-one column, used in Newton's method, false, do not add it, used in gradient descent
      val trainDataRDD: RDD[Instance] = df2RDD(trTrainDataDF, featuresVecName, "label", trainingMethod != "Gradient")

      val trainingStartTime = System.nanoTime()

      // gradient descent or Newton's method
      val (theta, intercept, lost): (Vector, Double, Array[Double]) = train(trainingMethod)(trainDataRDD, 100, 0.0001, 0.1)

      val trainingDuration = (System.nanoTime() - trainingStartTime) / 1e9d
      println("My Training duration: " + trainingDuration + " s.")

      println("My theta: " + theta.toArray.mkString(", "))
      println("My intercept: " + intercept)
      println("My lost: " + lost.mkString(", "))

      /* end train */

      /* begin test */

      // Used in gradient descent or Newton's method, the last parameter is false, gradient descent, true, Newton's method
      val testDataRDD: RDD[Instance] = df2RDD(trTestDataDF, featuresVecName, "label", trainingMethod != "Gradient")
      val (predictions, confusionMatrix) = predict(testDataRDD, theta, intercept)

      predictions.toDF("label", "predictionProb", "prediction").take(20).foreach(println)

      println("accuracy: " + score(confusionMatrix))

      /* end test */
    }

    if(displayCV) {
      /* begin cross validation */

      val lambdaDoubleParam: DoubleParam =
        new DoubleParam(Identifiable.randomUID("lambda"), "lambda", "lambda parameter (>=0.0)", ParamValidators.gtEq(0))
      val polyDegreeIntParam: IntParam =
        new IntParam(Identifiable.randomUID("polyDegree"), "polyDegree", "polynomial degree parameter (>=1)", ParamValidators.gtEq(1))

      val paramGrid: Array[ParamMap] = new ParamGridBuilder()
        .addGrid(lambdaDoubleParam, Array(0.01, 0.1))
        .addGrid(polyDegreeIntParam, Array(1, 3))
        .build()

      val trainingStartTime = System.nanoTime()

      val (bestTheta, bestIntercept, _, bestParaMap, _) =
        crossValidation(trainingMethod)(trTrainDataDF, featuresVecName, "label", 3)(paramGrid, lambdaDoubleParam, polyDegreeIntParam, "accuracy")(40, 0.01)

      val trainingDuration = (System.nanoTime() - trainingStartTime) / 1e9d
      println("Cross validation training duration: " + trainingDuration + " s.")

      println("bestParaMap: " + bestParaMap)

      /* end cross validation */


      /* begin test */

      // polynomial expansion
      val polyNomialExpansion: PolynomialExpansion = new PolynomialExpansion().setInputCol(featuresVecName)
        .setOutputCol("CVBestFeaturesVec").setDegree(bestParaMap.getOrElse[Int](polyDegreeIntParam, 1))

      // Used in gradient descent or Newton's method, the last parameter is false, gradient descent, true, Newton's method
      val testDataRDD: RDD[Instance] = df2RDD(polyNomialExpansion.transform(trTestDataDF),
        "CVBestFeaturesVec", "label", trainingMethod != "Gradient")

      // lazy prediction, be cautious
      val (predictions, confusionMatrix) = predict(testDataRDD, bestTheta, bestIntercept)

      predictions.toDF("label", "predictionProb", "prediction").sample(withReplacement = false, 0.0001).map{
        case Row(label, predictionProb, prediction) => label + " " + predictionProb + " " + prediction
      }.foreach(println(_))

      println("accuracy: " + score(confusionMatrix))

      /* end test */
    }

//    sparkSql.stop()
//    sc.stop()
  }

}

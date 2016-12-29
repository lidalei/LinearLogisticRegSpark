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
import hyperparameter.tuning.MyCrossValidation.kFold
import org.apache.spark.ml.param.{DoubleParam, IntParam, ParamMap, ParamValidators}
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.storage.StorageLevel

/**
  * Created by Sophie on 11/23/16.
  */


object MyLogisticRegression {

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
    * @param data
    * @param theta
    * @return theta and intercept
    */
  private def computeGradient(data: RDD[Instance], theta: Vector, intercept: Double): (Vector, Double) = {
    // compute gradient
    data.map({
      case Instance(label, features) => {
        val gradientFactor: Double = sigmoid(vecInnerProduct(theta, features) + intercept) - label
        // theta and intercept
        (vecScale(features, gradientFactor), gradientFactor)
      }
    }).reduce((pair1: (Vector, Double), pair2: (Vector, Double)) => (vecAdd(pair1._1, pair2._1), pair1._2 + pair2._2))
  }

  /**
    * @deprecated
    * Clip the value into (min, max) to avoid numerical instability
    * @param value
    * @param min
    * @param max
    * @return
    */
  private def clip(value: Double, min: Double, max: Double): Double = {
    if(value > max) {
      max
    }
    else if(value < min) {
      min
    }
    else {
      value
    }
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
    * @param data
    * @param theta
    * @return
    */
  private def crossEntropy(data: RDD[Instance], theta: Vector, intercept: Double): Double = {
    // numerical stability, be cautious with zero and one, using log sum trick
    // https://lingpipe-blog.com/2012/02/16/howprevent-overflow-underflow-logistic-regression/
    data.map({
      case Instance(label, features) => {
        val z: Double = vecInnerProduct(theta, features) + intercept

        if(label == 1.0) {
          log1pExp(-z)
        }
        else {
          log1pExp(z)
        }
      }
    }).mean()
  }


  /**
    * MyLogisticRegression train, Gradient descent or Newton's method
    * Somewhere the bias should be considered...
    * @param trainingMethod if gradient descent, intercept is fit separately. If Newton's, features contains all-one column
    * @param trainData
    * @param batchSize the number of batches, default 1, only applies on Gradient descent now
    * @param maxIterations
    * @param learningRate
    * @param lambda
    * @param tolerance
    * @return fit coefficients, theta plus intercept plus training losts
    */
  def train(trainingMethod: String = "Gradient", batchSize: Int = 1)
           (trainData: RDD[Instance], maxIterations: Int, learningRate: Double, lambda: Double, tolerance: Double = 1e-6): (Vector, Double, Array[Double]) = {

    require(trainingMethod.equals("Gradient") || trainingMethod.equals("Newton"))
    require(batchSize >= 1 && maxIterations >= 1 && lambda >= 0.0 && tolerance >= 0.0)

    val handlePersistence = trainData.getStorageLevel == StorageLevel.NONE
    if(handlePersistence) {
      trainData.persist(StorageLevel.MEMORY_AND_DISK)
    }

    val numberOfFeatures: Int = trainData.first() match {
      case Instance(_, features) => features.size
    }

    // set initial coefficients and intercept
    var theta: Vector = Vectors.zeros(numberOfFeatures)
    var intercept: Double = 0.0

    val lost = Array.fill[Double](maxIterations)(0.0)
    var decayedLearningRate = learningRate

    // TODO implement mini-batch or early stop at validation set
    var (i, delta): (Int, Double) = (0, 100.0)

    if(trainingMethod.equals("Gradient")) {
//      if(batchSize > 1) {
//        // divide data into batchSize parts
//
//      }
      while(i < maxIterations && delta > tolerance) {
        if(i >= 2 && lost(i - 1) > lost(i - 2)) {
          decayedLearningRate = decayedLearningRate / 2.0
        }

        // gradient of cross entropy at theta and intercept
        val (gradientOfThetaXE, gradientOfInterceptXE) = computeGradient(trainData, theta, intercept)

        // do not regularize intercept
        val gradientOfThetaRegularizer: Vector = vecScale(theta, lambda)

        // add l2 regularization. minus, because of "descent"
        val deltaTheta = vecScale(vecAdd(gradientOfThetaXE, gradientOfThetaRegularizer), -decayedLearningRate)
        theta = vecAdd(theta, deltaTheta)

        val deltaIntercept = -decayedLearningRate * gradientOfInterceptXE
        intercept = intercept + deltaIntercept
        lost(i) = crossEntropy(trainData, theta, intercept)

        i += 1
        // TODO, change of delta or change of cross entropy
        delta = math.sqrt(vecNormPower(deltaTheta, 2) + deltaIntercept * deltaIntercept)
      }
    }
    else {
      // keep intercept as zero
      intercept = 0.0

      while(i < maxIterations && delta > tolerance) {
        if(i >= 2 && lost(i - 1) > lost(i - 2)) {
          decayedLearningRate = decayedLearningRate / 2.0
        }
        val (gradientOfTheta, _) = computeGradient(trainData, theta, intercept)
        // compute Hessian matrix
        val hessianMatrixVec = computeHessianMatrix(trainData, theta)

        val eyeMatrixArr: Array[Double] = DenseMatrix.eye(numberOfFeatures).toArray

        // size = numberOfFeatures * numberOfFeatures
        val invHessianMatrixArr: Array[Double] = solveLinearEquations(hessianMatrixVec.toArray, eyeMatrixArr, numberOfFeatures, numberOfFeatures)

        val deltaNewton = Matrices.dense(numberOfFeatures, numberOfFeatures, invHessianMatrixArr).multiply(gradientOfTheta)

        val deltaTheta = vecScale(deltaNewton, -decayedLearningRate)

        theta = vecAdd(theta, deltaTheta)

        lost(i) = crossEntropy(trainData, theta, intercept)

        i += 1
        // TODO, change of delta or change of cross entropy
        delta = math.sqrt(vecNormPower(deltaTheta, 2))
      }
    }

    if(handlePersistence) {
      trainData.unpersist()
    }

    // remove un-computed lost if early stop
    (theta, intercept, lost.slice(0, i))
  }

  def computeHessianMatrix(data: RDD[Instance], theta: Vector): Vector = {
    data.map({
      case Instance(label: Double, features: Vector) => {
        val sigma = sigmoid(vecInnerProduct(theta, features))
        vecScale(outerVecProduct(features), sigma * (1 - sigma))
      }
    }).reduce(vecAdd)
  }


  /**
    * Make predictions and compute classification metrics
    * @param testData
    * @param threshold
    * @return
    */
  def predict(testData: RDD[Instance], theta: Vector, intercept: Double, threshold: Double = 0.5): (RDD[InstanceWithPrediction], ConfusionMatrix) = {
    require(threshold > 0 && threshold < 1)

    // make predictions, lazy operation, be cautious, accumulator here might be updated more times due to failure of tasks
    val predictions: RDD[InstanceWithPrediction] = testData.map{
      case Instance(label, features) => {
        val predictionProb = sigmoid(vecInnerProduct(theta, features) + intercept)
        val prediction: Double = if(predictionProb >= threshold) 1.0 else 0.0

        InstanceWithPrediction(label, features, predictionProb, prediction)
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
      case InstanceWithPrediction(label, _, _, prediction) => {
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
    * Cross validation to do hyperparameter tuning for logistic regression
    * @param dataDF
    * @param featuresVecName
    * @param label
    * @param paramGrid
    * @param maxIterations
    * @param learningRate
    * @param threshold
    * @return
    */
  def crossValidation(dataDF: DataFrame, featuresVecName: String, label: String = "label", k: Int = 3)
                     (paramGrid: Array[ParamMap], lambdaDoubleParam: DoubleParam, polyDegreeIntParam: IntParam, scoreType: String)
                     (maxIterations: Int, learningRate: Double, threshold: Double = 0.5): Array[(ParamMap, Double)] = {

    // search over the hyper-parameter grid spanned by l2 penalty and polynomial expansion
    // outer loop - grid search
    var polyDegreesSet: Set[Int] = Set[Int]()
    var lambdasSet: Set[Double] = Set[Double]()

    paramGrid.foreach((paramMap: ParamMap) => {
      val lambda: Double = paramMap.getOrElse[Double](lambdaDoubleParam, 0.0)
      val polyDegree: Int = paramMap.getOrElse[Int](polyDegreeIntParam, 1)

      polyDegreesSet += polyDegree
      lambdasSet += lambda
    })

    val polyDegrees: Array[Int] = polyDegreesSet.toArray
    val lambdas: Array[Double] = lambdasSet.toArray

    // TODO, avoid unnecessary data pass (transform data only the number of polyDegree times)
    val cvScores: Array[(Int, Double, Double)] = polyDegrees.map((polyDegree: Int) => {
      // polynomial expansion
      val polyNomialExpansion: PolynomialExpansion = new PolynomialExpansion().setInputCol(featuresVecName)
        .setOutputCol("CVFeaturesVec").setDegree(polyDegree)

      val dataRDD: RDD[Instance] = df2RDD(polyNomialExpansion.transform(dataDF), "CVFeaturesVec", label, false)

      // inner loop - cross validation
      val kFolds: Array[(RDD[Instance], RDD[Instance])] = kFold(dataRDD, k)

      val cvpolyDegreeScores: Array[Array[(Int, Double, Double)]] = kFolds.map{
        case (trainData: RDD[Instance], testData: RDD[Instance]) => {
          trainData.persist()
          testData.persist()

          // add lambda
          val polyDegreeLambdasScores: Array[(Int, Double, Double)] = lambdas.map((lambda: Double) => {
            val (theta: Vector, intercept: Double, lost: Array[Double]) = train("Gradient")(trainData, maxIterations, learningRate, lambda)
            val (predictions: RDD[InstanceWithPrediction], confusionMatrix: ConfusionMatrix) = predict(testData, theta, intercept, threshold)

            (polyDegree, lambda, score(confusionMatrix, scoreType))
          })

          trainData.unpersist()
          testData.unpersist()

          polyDegreeLambdasScores
        }
      }

      cvpolyDegreeScores
    }).fold(Array[Array[(Int, Double, Double)]]())(_ ++ _).fold(Array[(Int, Double, Double)]())(_ ++ _)

    // TODO, compute mean cv scores

    paramGrid.map((paramMap: ParamMap) => {
      val lambda: Double = paramMap.getOrElse[Double](lambdaDoubleParam, 0.0)
      val polyDegree: Int = paramMap.getOrElse[Int](polyDegreeIntParam, 1)

      // polynomial expansion
      val polyNomialExpansion: PolynomialExpansion = new PolynomialExpansion().setInputCol(featuresVecName)
        .setOutputCol("featuresVec").setDegree(polyDegree)

      val dataRDD: RDD[Instance] = df2RDD(polyNomialExpansion.transform(dataDF), "featuresVec", label, false)

      // inner loop - cross validation
      val kFolds: Array[(RDD[Instance], RDD[Instance])] = kFold(dataRDD, k)
      val meanScore: Double = kFolds.map{
        case (trainData: RDD[Instance], testData: RDD[Instance]) => {
          // add lambda
          val (theta: Vector, intercept: Double, lost: Array[Double]) = train("Gradient")(trainData, maxIterations, learningRate, lambda)
          val (predictions: RDD[InstanceWithPrediction], confusionMatrix: ConfusionMatrix) = predict(testData, theta, intercept, threshold)
          score(confusionMatrix, scoreType)
      }}.sum / k

      (paramMap, meanScore)
    })
  }


  /**
    * Starting point of the program
    * @param args, args(0) is the path to the dataset
    */
  def main(args: Array[String]): Unit = {
    // whether to display the result of mllib logistic regression
    val displayMLLogReg: Boolean = false

    // Gradient or Newton
    val trainingMethod: String = "Gradient"

    // Do not use all the cores. TODO, comment setMaster to run in a cluster
    val conf = new SparkConf()
      .setAppName("My Logistic Regression").set("spark.executor.cores", "6")
      .setMaster("local[2]")

    val sc = initializeSC(conf)
    val sparkSql = initializeSparkSession(conf)

    sc.setLogLevel("WARN")

    import sparkSql.implicits._
    import sparkSql._

    // check the contents of dataset
    var filePath = "/Users/Sophie/Downloads/MPML-Datasets/HIGGS_sample.csv"

    // the same file should be in all nodes of a cluster, including the master and workers
    if(args.length >= 1) {
      filePath = args(0)
    }

    // read data
//    val dataRDD = sc.textFile(filePath)
//    dataRDD.take(1000).foreach(println)

    val featuresCols: Array[String] = (1 to 28).toArray.map("col" + _)
    val cols: Array[String] = Array("label") ++ featuresCols
    val fields: Array[StructField] = cols.map(StructField(_, DoubleType, nullable = true))
    val schema = StructType(fields)
    // persist data
    val dataDF = sparkSql.read.option("header", false).schema(schema).csv(filePath)


    // split data into train and test, TODO, change train / test to speed up
    val trainTestSplitArr: Array[DataFrame] = dataDF.randomSplit(Array(0.7, 0.3))
    val (trainDataDF, testDataDF) = (trainTestSplitArr(0), trainTestSplitArr(1))

    /* begin data preparation */

    // 1. form a vector of all level features
    val allFeaturesVecAssembler = new VectorAssembler().setInputCols(featuresCols).setOutputCol("allFeaturesVec")

    // 2. form a vector of all low level features
    val lowLevelFeaturesCols: Array[String] = cols.slice(1, 22)
    val lowLevelFeaturesVecAssembler = new VectorAssembler().setInputCols(lowLevelFeaturesCols).setOutputCol("lowLevelFeaturesVec")

    // 3. form a vector of all high level features
    val highLevelFeaturesCols: Array[String] = cols.slice(22, 29)
    val highLevelFeaturesVecAssembler = new VectorAssembler().setInputCols(highLevelFeaturesCols).setOutputCol("highLevelFeaturesVec")

    // 4. logistic regression within Spark ml, TODO, only use low level features now.
    val featuresVecName: String = "lowLevelFeaturesVec"
    // transform train and test data, TODO, only use low level features now.
    val trTrainDataDF = lowLevelFeaturesVecAssembler.transform(trainDataDF)
    val trTestDataDF = lowLevelFeaturesVecAssembler.transform(testDataDF)

    /* end data preparation */

    /* begin train */
    // mllib logistic regression, very similar with Newton's method
    if(displayMLLogReg) {
      val trainingStartTime = System.nanoTime()

      val logisticRegression: LogisticRegression = new LogisticRegression().setLabelCol("label").setFeaturesCol(featuresVecName).setMaxIter(100).setFitIntercept(true)
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

    // My Logistic Regression implementations

    // if the last parameter is true, add all-one column, used in Newton's method, false, do not add it, used in gradient descent
    val trainDataRDD: RDD[Instance] = df2RDD(trTrainDataDF, featuresVecName, "label", !trainingMethod.equals("Gradient"))

    val trainingStartTime = System.nanoTime()

    // gradient descent or Newton's method
    val (theta, intercept, lost): (Vector, Double, Array[Double]) = train(trainingMethod)(trainDataRDD, 100, 0.0001, 0.1)

    val trainingDuration = (System.nanoTime() - trainingStartTime) / 1e9d
    println("Training duration: " + trainingDuration + " s.")

    println("theta: " + theta.toArray.mkString(", "))
    println("intercept: " + intercept)
    println("lost: " + lost.mkString(", "))

    /* end train */

    /* begin test */

    // Used in gradient descent or Newton's method, the last parameter is false, gradient descent, true, Newton's method
    val testDataRDD: RDD[Instance] = df2RDD(trTestDataDF, featuresVecName, "label", !trainingMethod.equals("Gradient"))

    // lazy prediction, be cautious
    val (predictions, confusionMatrix) = predict(testDataRDD, theta, intercept, 0.5)

    predictions.toDF("label", "features", "predictionProb", "prediction").select("label", "predictionProb", "prediction").take(20).foreach(println)

    println("accuracy: " + score(confusionMatrix))

    /* end test */

    val lambdaDoubleParam: DoubleParam = new DoubleParam(Identifiable.randomUID("lambdaDoubleParam"), "lambdaDoubleParam", "lambda parameter (>=0.0)", ParamValidators.gtEq(0))
    val polyDegreeIntParam: IntParam = new IntParam(Identifiable.randomUID("polyDegreeParam"), "polyDegreeParam", "polynomial degree parameter (>=1)", ParamValidators.gtEq(1))

    val paramGrid: Array[ParamMap] = new ParamGridBuilder()
      .addGrid(lambdaDoubleParam, Array(0.001, 0.01, 0.1, 0.5))
      .addGrid(polyDegreeIntParam, Array(1, 2, 3))
      .build()

    val cvRes = crossValidation(trTrainDataDF, featuresVecName, "label", 3)(paramGrid, lambdaDoubleParam, polyDegreeIntParam, "accuracy")(40, 0.01, 0.5)

    cvRes.map(println)


    /* test */


    sc.stop()
  }

}

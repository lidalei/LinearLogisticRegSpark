package classification

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.types.{DoubleType, StructField, StructType}
import org.apache.spark.ml.linalg.{DenseMatrix, Matrices, Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions.col
import Helper.VectorMatrixManipulation._
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.util.LongAccumulator

/**
  * Created by Sophie on 11/23/16.
  */


case class Instance(label: Double, features: Vector)

/**
  * The case class to hold Instance and its predictionProb to be class 1
  * @param label: true class
  * @param features: features used to predict
  * @param predictionProb: probability to be class 1
  */
case class InstanceWithPredictionProb(label: Double, features: Vector, predictionProb: Double)
case class InstanceWithPrediction(label: Double, features: Vector, predictionLabel: Double)

case class ConfusionMatrix(truePositive: LongAccumulator, trueNegative: LongAccumulator, falsePositive: LongAccumulator, falseNegative: LongAccumulator) {
  override def toString: String = {
    "truePositive: " + truePositive.value + ", trueNegative: " + trueNegative.value + ", falsePositive: " + falsePositive.value + ", falseNegative: " + falseNegative.value
  }

}
//case class ClassificationMetrics(accuracy: Double, precision: Double, recall: Double)



object MyLogisticRegression {

  def initializeSC(): SparkContext = {
    val conf = new SparkConf().setAppName("My Logistic Regression").setMaster("local[2]")
    val sc = new SparkContext(conf)
    sc
  }

  def initializeSparkSqlSession(): SparkSession = {
    val sparkSql = SparkSession
      .builder().master("local[2]")
      .appName("My Logistic Regression")
      //      .config("spark.some.config.option", "some-value")
      .getOrCreate()
    sparkSql
  }

  def sigmoid(z: Double): Double  ={
    1.0 / (1.0 + math.exp(-z))
  }

  def computeGradient(data: RDD[Instance], theta: Vector): Vector = {
    // compute gradient, to implement newton's method and mini-batch
    data.map({
      case Instance(label, features) => vecScale(features, sigmoid(vecInnerProduct(theta, features)) - label)
    }).reduce(vecAdd)
  }

  def crossEntropy(data: RDD[Instance], theta: Vector): Double = {

    // if not use sigmoid, to be cautious with zero
    data.map({
      case Instance(label, features) => {
        val p = sigmoid(vecInnerProduct(theta, features))
        if(label == 1.0) {
          -math.log(p)
        }
        else {
          -math.log(1.0 - p)
        }
      }
    }).mean()
  }

  /**
    * MyLogisticRegression train
    * Somewhere the bias should be considered...
    * @param trainData
    * @return fit coefficients
    */

  def trainGradientDescent(trainData: RDD[Instance], maxIterations: Int, learningRate: Double): (Vector, Array[Double]) = {

    val numberOfFeatures: Int = trainData.first() match {
      case Instance(_, features) => features.size
    }

    // set initial coefficients, with all one column
    var theta: Vector = Vectors.zeros(numberOfFeatures)
    val lost = Array.fill[Double](maxIterations)(0.0)
    var decayedLearningRate = learningRate

    for {
      i <- 0 until maxIterations
    }
      {
        if(i >= 2 && lost(i - 1) > lost(i - 2)) {
          decayedLearningRate = learningRate / 2.0
        }
        // minus
        val deltaTheta = vecScale(computeGradient(trainData, theta), -decayedLearningRate)
        theta = vecAdd(theta, deltaTheta)

        lost(i) = crossEntropy(trainData, theta)
      }

    (theta, lost)
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
    * train with Newton's method
    * @param trainData
    * @param maxIterations
    * @param learningRate
    * @return
    */
  def trainNewtonMethod(trainData: RDD[Instance], maxIterations: Int, learningRate: Double): (Vector, Array[Double]) = {
    val numberOfFeatures: Int = trainData.first() match {
      case Instance(_, features) => features.size
    }

    // set initial coefficients, with all one column
    var theta: Vector = Vectors.zeros(numberOfFeatures)
    val lost = Array.fill[Double](maxIterations)(0.0)
    var decayedLearningRate = learningRate

    for {
      i <- 0 until maxIterations
    }
      {
        if(i >= 2 && lost(i - 1) > lost(i - 2)) {
          decayedLearningRate = learningRate / 2.0
        }
        val gradient = computeGradient(trainData, theta)
        // compute Hessian matrix
        val hessianMatrixVec = computeHessianMatrix(trainData, theta)

        val eyeMatrixArr: Array[Double] = DenseMatrix.eye(numberOfFeatures).toArray

        // size = numberOfFeatures * numberOfFeatures
        val invHessianMatrixArr: Array[Double] = solveLinearEquations(hessianMatrixVec.toArray, eyeMatrixArr, numberOfFeatures, numberOfFeatures)

        val deltaNewton = Matrices.dense(numberOfFeatures, numberOfFeatures, invHessianMatrixArr).multiply(gradient)

        val deltaTheta = vecScale(deltaNewton, -decayedLearningRate)

        theta = vecAdd(theta, deltaTheta)

        lost(i) = crossEntropy(trainData, theta)
      }

    (theta, lost)
  }

  /**
    * MyLogisticRegression predict
    * @param theta
    * @param testData
    * @return rdd with predicted
    */
  def predictProb(theta: Vector, testData: RDD[Instance]): RDD[InstanceWithPredictionProb] = {
    testData.map({
      case Instance(label, features) => InstanceWithPredictionProb(label, features, sigmoid(vecInnerProduct(theta, features)))
    })
  }


  /**
    * Make predictions and compute classification metrics
    * @param testData
    * @param threshold
    * @return
    */
  def predict(testData: RDD[InstanceWithPredictionProb], threshold: Double = 0.5): (RDD[InstanceWithPrediction], ConfusionMatrix) = {
    require(threshold > 0 && threshold < 1)

    val sc: SparkContext = SparkContext.getOrCreate()

    // accumulators to store confusion matrix
    val truePositiveAcc: LongAccumulator = sc.longAccumulator("truePositive")
    val falsePositiveAcc: LongAccumulator = sc.longAccumulator("falsePositive")
    val trueNegativeAcc: LongAccumulator = sc.longAccumulator("trueNegative")
    val falseNegativeAcc: LongAccumulator = sc.longAccumulator("falseNegative")

    // make predictions
    val predictions: RDD[InstanceWithPrediction] = testData.map({
      case InstanceWithPredictionProb(label, features, predictionProb) => {
        val prediction: Double = if(predictionProb >= threshold) 1.0 else 0.0

        if(prediction == 1.0 && label == 1.0) {
          truePositiveAcc.add(1)
        }
        else if(prediction == 1.0 && label == 0.0) {
          falsePositiveAcc.add(1)
        }
        else if(prediction == 0.0 && label == 1.0) {
          falseNegativeAcc.add(1)
        }
        else {
          trueNegativeAcc.add(1)
        }

        InstanceWithPrediction(label, features, prediction)
      }
    })

    // only driver program can get the value
    // println("truePositive value" + truePositiveAcc.value) will print 0
    (predictions, ConfusionMatrix(truePositiveAcc, trueNegativeAcc, falsePositiveAcc, falseNegativeAcc))

  }

  def main(args: Array[String]): Unit = {

    val sc = initializeSC()
    val sparkSql = initializeSparkSqlSession()

    import sparkSql.implicits._
    import sparkSql._

    // check the contents of dataset
    val filePath = "src/main/resources/HIGGS_sample.csv"

//    val dataRDD = sc.textFile(filePath)
//    dataRDD.take(1000).foreach(println)

    val featuresCols: Array[String] = (1 to 28).toArray.map("col" + _)
    val cols: Array[String] = Array("label") ++ featuresCols
    val fields: Array[StructField] = cols.map(StructField(_, DoubleType, nullable = true))
    val schema = StructType(fields)
    val dataDF = sparkSql.read.option("header", false).schema(schema).csv(filePath)


    // split data into train and test, TODO
    val trainTestSplitArr: Array[DataFrame] = dataDF.randomSplit(Array(0.8, 0.2))
    val (trainDataDF, testDataDF) = (trainTestSplitArr(0), trainTestSplitArr(1))

    // 1. form a vector of all level features
    val lowLevelFeaturesCols: Array[String] = cols.slice(1, 22)
    val lowLevelFeaturesVecAssembler = new VectorAssembler().setInputCols(lowLevelFeaturesCols).setOutputCol("lowLevelFeaturesVec")

    // 2. form a vector of all high level features
    val highLevelFeaturesCols: Array[String] = cols.slice(22, 29)
    val highLevelFeaturesVecAssembler = new VectorAssembler().setInputCols(highLevelFeaturesCols).setOutputCol("highLevelFeaturesVec")

    // 3. form a vector of all features
    val allFeaturesVecAssembler = new VectorAssembler().setInputCols(featuresCols).setOutputCol("allFeaturesVec")

    // 4. logistic regression within Spark ml
    val featuresVecName: String = "highLevelFeaturesVec"
    val logisticRegression = new LogisticRegression("label").setFeaturesCol(featuresVecName).setMaxIter(10).setFitIntercept(true)

    val transformStages = Array(highLevelFeaturesVecAssembler, logisticRegression)

    val pipeline = new Pipeline().setStages(transformStages)

    val pipelineModel = pipeline.fit(trainDataDF)

    val mlLogisticRegModel = pipelineModel.stages(transformStages.length - 1).asInstanceOf[LogisticRegressionModel]
    val mlTheta: Vector = mlLogisticRegModel.coefficients

    println("ML Theta: " + mlTheta.toArray.mkString(", "))
    println("ML Objective history: " + mlLogisticRegModel.summary.objectiveHistory.mkString(", "))

    val trTrainDataDF = pipelineModel.transform(trainDataDF)

    val trTestDataDF = pipelineModel.transform(testDataDF)


//    trTrainDataDF.printSchema()
    // very costy
//    trTrainDataDF.describe(cols: _*).show()
//    trTrainDataDF.show()

    // persist trainDataRDD
    val trainDataRDD: RDD[Instance] = trTrainDataDF.select(col("label"), col(featuresVecName)).rdd.map({
      case Row(label: Double, features: Vector) => Instance(label, Vectors.dense(features.toArray ++ Array(1.0)))
    }).persist()


    val testDataRDD: RDD[Instance] = trTestDataDF.select(col("label"), col(featuresVecName)).rdd.map({
      case Row(label: Double, features: Vector) => Instance(label, Vectors.dense(features.toArray ++ Array(1.0)))
    })

//    val (theta, lost): (Vector, Array[Double]) = trainGradientDescent(trainDataRDD, 40, 0.1)

    val (theta, lost): (Vector, Array[Double]) = trainNewtonMethod(trainDataRDD, 40, 0.1)
    // unpersist trainDataRDD
    trainDataRDD.unpersist()

    println("theta: " + theta.toArray.mkString(", "))
    println("lost: " + lost.mkString(", "))

    // .toDF("label", "features", "predictionProb")
    val predictionProbs: RDD[InstanceWithPredictionProb] = predictProb(theta, testDataRDD)

    predictionProbs.toDF().select("label", "predictionProb").show()

    // lazy prediction, be cautious
    val (predictions, confusionMatrix) = predict(predictionProbs, 0.5)

    // force to predict all
    predictions.count()

    predictions.toDF("label", "features", "predictionLabel").select("label", "predictionLabel").show()

    println(confusionMatrix.toString)

  }

}

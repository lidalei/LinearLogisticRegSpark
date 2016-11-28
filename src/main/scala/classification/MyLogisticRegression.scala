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

/**
  * Created by Sophie on 11/23/16.
  */


case class Instance(label: Double, features: Vector)
case class InstanceWithPrediction(label: Double, features: Vector, prediction: Double)


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
    }).sum()
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
  def predict(theta: Vector, testData: RDD[Instance]): RDD[InstanceWithPrediction] = {
    testData.map({
      case Instance(label, features) => InstanceWithPrediction(label, features, sigmoid(vecInnerProduct(theta, features)))
    })
  }


  def main(args: Array[String]): Unit = {

    val sc = initializeSC()
    val sparkSql = initializeSparkSqlSession()

    import sparkSql.implicits._
    import sparkSql._

    // check the contents of dataset
    val filePath = "src/main/resources/HIGGS_sample.csv"

//    val dataRDD = sc.textFile(filePath)
//    dataRDD.take(200).foreach(println)

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
    val featuresVecName: String = "lowLevelFeaturesVec"
    val logisticRegression = new LogisticRegression("label").setFeaturesCol(featuresVecName).setMaxIter(10).setFitIntercept(true)

    val transformStages = Array(lowLevelFeaturesVecAssembler, logisticRegression)

    val pipeline = new Pipeline().setStages(transformStages)

    val pipelineModel = pipeline.fit(trainDataDF)

    val mlTheta: Vector = pipelineModel.stages(transformStages.length - 1).asInstanceOf[LogisticRegressionModel].coefficients
    println(mlTheta.toArray.mkString(","))

    val trTrainDataDF = pipelineModel.transform(trainDataDF)

    val trTestDataDF = pipelineModel.transform(testDataDF)


//    trTrainDataDF.printSchema()
    // very costy
//    trTrainDataDF.describe(cols: _*).show()
//    trTrainDataDF.show()


    val trainDataRDD: RDD[Instance] = trTrainDataDF.select(col("label"), col(featuresVecName)).rdd.map({
      case Row(label: Double, features: Vector) => Instance(label, Vectors.dense(features.toArray ++ Array(1.0)))
    })

    val testDataRDD: RDD[Instance] = trTestDataDF.select(col("label"), col(featuresVecName)).rdd.map({
      case Row(label: Double, features: Vector) => Instance(label, Vectors.dense(features.toArray ++ Array(1.0)))
    })

//    val (theta, lost): (Vector, Array[Double]) = trainGradientDescent(trainDataRDD, 40, 0.01)

    val (theta, lost): (Vector, Array[Double]) = trainNewtonMethod(trainDataRDD, 40, 0.01)

    println("theta: " + theta.toArray.mkString(","))
    println("lost: " + lost.mkString(","))

    val predictionProbs = predict(theta, testDataRDD).toDF("label", "features", "prediction")
    predictionProbs.select("label", "prediction").show()


  }





}

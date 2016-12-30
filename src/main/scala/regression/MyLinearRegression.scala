package regression

/**
  * Created by Sophie on 11/23/16.
  */
import Helper.{Array2VectorUDF, Vector2DoubleUDF}
import Helper.VectorMatrixManipulation._
import org.apache.spark.ml.feature.{RegexTokenizer, StandardScaler, StandardScalerModel, VectorSlicer}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.{DataFrame, Row}
import Helper.InstanceUtilities.{Instance, InstanceWithPredictionReg, initializeSC, initializeSparkSession}
import org.apache.spark.SparkConf

object MyLinearRegression {

  def train(trainInstances: RDD[Instance], l2Regularization: Double): Vector = {
    val normalEquTerms: (Vector, Vector) = trainInstances.map((instance: Instance) => instance match {
      case Instance(aLabel, aFeatures) => (outerVecProduct(aFeatures), vecScale(aFeatures, aLabel))
    }).reduce((pair1: (Vector, Vector), pair2: (Vector, Vector)) => (vecAdd(pair1._1, pair2._1), vecAdd(pair1._2, pair2._2)))

    val normalEquTerm2: Vector = normalEquTerms._2
    val normalEquTerm1: Vector = normalEquTerms._1
    //    println(normalEquTerm2)
    //    println(normalEquTerm1)

    val regularizedNormalEquTerm1 = vecAdd(normalEquTerm1, upperTriangle(normalEquTerm2.size, l2Regularization))
    //    println(regularizedNormalEquTerm1)

    Vectors.dense(solve(regularizedNormalEquTerm1.toArray, normalEquTerm2.toArray))
  }


  def predict(theta: Vector, instances: RDD[Instance]): RDD[InstanceWithPredictionReg] = {
    instances.map({
      case Instance(label, features) => InstanceWithPredictionReg(label, features, vecInnerProduct(theta, features))
    })
  }

  /**
    * Compute RMSE
    */
  def rmse(predictions: RDD[InstanceWithPredictionReg]): Double = {
    math.sqrt(predictions.map({
      case InstanceWithPredictionReg(label, features, prediction) => (label - prediction) * (label - prediction)
    }).sum() / predictions.count())
  }

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAppName("My Linear Regression").setMaster("local[2]")
    val sc = initializeSC(conf)
    val sparkSql = initializeSparkSession(conf)

    import sparkSql.implicits._
    import sparkSql._

    // read dataset

    val filePath = "/Users/Sophie/Downloads/MPML-Datasets/YearPredictionMSD.txt"

    // check the content
//    val rdd = sc.textFile(filePath)
//    rdd.take(5).foreach(println)

    // read data as a dataframe
    val data: DataFrame = sparkSql.read.text(filePath)

    // split data into train and test
    val trainTestSplitArr: Array[DataFrame] = data.randomSplit(Array(0.7, 0.3))
    val (trainData, testData) = (trainTestSplitArr(0), trainTestSplitArr(1))

    val l2Regularization: Double = 0.1

    // begin the pipeline
    // 1. split by ,
    val splitter = new RegexTokenizer().setInputCol("value").setOutputCol("valueArr").setPattern(",")

    // 2. Array[String] to Vector[Double]
    val arr2Vec = new Array2VectorUDF(_.toDouble).setInputCol("valueArr").setOutputCol("valueVec")

    // 3. center the data, important!!!
    val stdScaler = new StandardScaler().setInputCol("valueVec").setOutputCol("valueVecCentered").setWithMean(true).setWithStd(false)

    // 4. slice year (label) out, as Vector[Double]
    val yearVecSlicer = new VectorSlicer().setInputCol("valueVecCentered").setOutputCol("yearVec").setIndices(Array(0))

    // 5. transform year to Double
    val yearVec2Double = new Vector2DoubleUDF(_(0)).setInputCol("yearVec").setOutputCol("label")

    // 6. slice year features out, as Vector[Double]
    val featuresVecSlicer = new VectorSlicer().setInputCol("valueVecCentered").setOutputCol("featuresVec").setIndices((1 to 90).toArray)

    // 7. split features into timbre average and timbre covariance
    val timbreAvgVecSlicer = new VectorSlicer().setInputCol("featuresVec").setOutputCol("timbreAvgVec").setIndices((0 to 11).toArray)
    val timbreCovVecSlicer = new VectorSlicer().setInputCol("featuresVec").setOutputCol("timbreCovVec").setIndices((12 to 89).toArray)

    val linearReg = new LinearRegression().setFeaturesCol("featuresVec").setLabelCol("label").setSolver("normal").setRegParam(l2Regularization)

    val transformStages = Array(splitter, arr2Vec, stdScaler, yearVecSlicer, yearVec2Double,  featuresVecSlicer, timbreAvgVecSlicer, timbreCovVecSlicer, linearReg)
    val transformPipeline = new Pipeline().setStages(transformStages)
    val pipelineModel:PipelineModel = transformPipeline.fit(trainData)

    val meanValues = pipelineModel.stages(2).asInstanceOf[StandardScalerModel].mean
    println("meanValues: " + meanValues)

    val trainingSummary = pipelineModel.stages(transformStages.length - 1).asInstanceOf[LinearRegressionModel].summary
    println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")

    val trainingCoefficients = pipelineModel.stages(transformStages.length - 1).asInstanceOf[LinearRegressionModel].coefficients
    println("Spark ML fit theta: " + trainingCoefficients.toArray.mkString(", "))

    // make predictions in test data
    val transformedTestData = pipelineModel.transform(testData)
    val linearRegTestRMSE = rmse(transformedTestData.select(col("label"), col("featuresVec"), col("prediction")).rdd.map({
      case Row(label: Double, features: Vector, prediction: Double) => InstanceWithPredictionReg(label, features, prediction)
    }))

    println("linearRegTestRMSE: " + linearRegTestRMSE)
    transformedTestData.select("label", "prediction").show()

    // implement my linear regression
    val transformedTrainData = pipelineModel.transform(trainData)

    val trainInstances: RDD[Instance] = transformedTrainData.select(col("label"), col("featuresVec")).rdd.map({
      case Row(label: Double, features: Vector) => Instance(label, features)
    })

    val testInstances: RDD[Instance] = transformedTestData.select(col("label"), col("featuresVec")).rdd.map({
      case Row(label: Double, features: Vector) => Instance(label, features)
    })

    val myTheta: Vector = train(trainInstances, l2Regularization)
    println("myTheta: " + myTheta.toArray.mkString(", "))

    // make predictions on test data
    val predictions: RDD[InstanceWithPredictionReg] = predict(myTheta, testInstances)

    val myRMSE = rmse(predictions)
    println("myRMSE: " + myRMSE)

    val predictionsDF = predictions.toDF("label", "features", "prediction")
    predictionsDF.select("label", "prediction").show()


  }
}
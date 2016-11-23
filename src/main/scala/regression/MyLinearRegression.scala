package regression

/**
  * Created by Sophie on 11/23/16.
  */

import Helper.{Array2VectorUDF, Vector2DoubleUDF}
import Helper.VectorManipulation._
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{DataFrame, SparkSession}


object MyLinearRegression {

  def initializeSC(): SparkContext = {
    val conf = new SparkConf().setAppName("My Linear Regression").setMaster("local[2]")
    val sc = new SparkContext(conf)
    sc
  }

  def initializeSparkSqlSession(): SparkSession = {
    val sparkSql = SparkSession
      .builder().master("local[2]")
      .appName("My Linear Regression")
      //      .config("spark.some.config.option", "some-value")
      .getOrCreate()
    sparkSql
  }

  def main(args: Array[String]): Unit = {

    val sc = initializeSC()
    val sparkSql = initializeSparkSqlSession()

    import sparkSql.implicits._
    import sparkSql._

    // read dataset

    val filePath = "src/main/resources/YearPredictionMSD.txt"

    // check the content
//    val rdd = sc.textFile(filePath)
//    rdd.take(5).foreach(println)

    // read data as a dataframe
    val data: DataFrame = sparkSql.read.text(filePath)

    // split data into train and test, TODO, to be changed
    val trainTestSplitArr: Array[DataFrame] = data.randomSplit(Array(0.9, 0.1))
    val (trainData, testData) = (trainTestSplitArr(0), trainTestSplitArr(1))


    // begin the pipeline
    // 1. split by ,
    val splitter = new RegexTokenizer().setInputCol("value").setOutputCol("valueArr").setPattern(",")

    // 2. Array[String] to Vector[Double]
    val arr2Vec = new Array2VectorUDF(_.toDouble).setInputCol("valueArr").setOutputCol("valueVec")

    // 2. slice year (label) out, as Vector[Double]
    val yearVecSlicer = new VectorSlicer().setInputCol("valueVec").setOutputCol("yearVec").setIndices(Array(0))

    // 3. transform year to Double and subtract with minimum year, 1922
    val yearVec2Double = new Vector2DoubleUDF(_(0) - 1922.0).setInputCol("yearVec").setOutputCol("label")

    // 4. slice year features out, as Vector[Double]
    val featuresVecSlicer = new VectorSlicer().setInputCol("valueVec").setOutputCol("featuresVec").setIndices((1 to 90).toArray)

    // 5. min-max transform features
    val minMaxScaler = new MinMaxScaler().setInputCol("featuresVec").setOutputCol("featuresVecScaled")

    // 6. split features into timbre average and timbre covariance
    val timbreAvgVecSlicer = new VectorSlicer().setInputCol("featuresVecScaled").setOutputCol("timbreAvgVec").setIndices((0 to 11).toArray)
    val timbreCovVecSlicer = new VectorSlicer().setInputCol("featuresVecScaled").setOutputCol("timbreCovVec").setIndices((12 to 89).toArray)

    // implement my linear regression


    /*
    val linearReg = new LinearRegression().setFeaturesCol("featuresVecScaled").setLabelCol("label")

    val transformStages = Array(splitter, arr2Vec, yearVecSlicer, yearVec2Double,  featuresVecSlicer, minMaxScaler, timbreAvgVecSlicer, timbreCovVecSlicer, linearReg)
    val transformPipeline = new Pipeline().setStages(transformStages)

    val predictions = transformPipeline.fit(trainData).transform(testData)
    predictions.printSchema()
    predictions.show()
    */


  }
}
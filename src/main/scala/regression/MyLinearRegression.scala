package regression

/**
  * Created by Sophie on 11/23/16.
  */

import Helper.{Array2VectorUDF, Vector2DoubleUDF}
import Helper.VectorMatrixManipulation._
import org.apache.spark.ml.feature.{RegexTokenizer, StandardScaler, StandardScalerModel, VectorSlicer}
import org.apache.spark.ml.linalg.{DenseMatrix, Vector}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions.col
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}


case class Instance(label: Double, features: Vector)

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

    val linearReg = new LinearRegression().setFeaturesCol("featuresVec").setLabelCol("label").setSolver("normal").setRegParam(0.1)

    val transformStages = Array(splitter, arr2Vec, stdScaler, yearVecSlicer, yearVec2Double,  featuresVecSlicer, timbreAvgVecSlicer, timbreCovVecSlicer, linearReg)
    val transformPipeline = new Pipeline().setStages(transformStages)
    val pipelineModel = transformPipeline.fit(trainData)

    val meanValues = pipelineModel.stages(2).asInstanceOf[StandardScalerModel].mean
    println("meanValues: " + meanValues)
    val stdValues = pipelineModel.stages(2).asInstanceOf[StandardScalerModel].std
    println("stdValues: " + stdValues)

//    val trainingSummary = pipelineModel.stages(transformStages.length - 1).asInstanceOf[LinearRegressionModel].summary
//    println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")

    //    val predictions = pipelineModel.transform(testData)
    //    predictions.printSchema()
    //    predictions.show()

    // implement my linear regression
    val transformedData = pipelineModel.transform(trainData)

    val l2Regularization: Double = 0.1

    val instances: RDD[Instance] = transformedData.select(col("label"), col("featuresVec")).rdd.map({
      case Row(label: Double, features: Vector) => Instance(label, features)
    })

    val normalEquTerms: (DenseMatrix, Vector) = instances.map((instance: Instance) => instance match {
      case Instance(aLabel, aFeatures) => (outerVecProduct(aFeatures, aFeatures), vecScale(aFeatures, aLabel))
    }).reduce((pair1: (DenseMatrix, Vector), pair2: (DenseMatrix, Vector)) => (matrixAdd(pair1._1, pair2._1), vecAdd(pair1._2, pair2._2)))

    val normalEquTerm1 = normalEquTerms._1
    val regularizedNormalEquTerm1 = matrixAdd(normalEquTerm1, matrixScale(DenseMatrix.ones(normalEquTerm1.numRows, normalEquTerm1.numCols), l2Regularization))
    val normalEquTerm2 = normalEquTerms._2


    println(regularizedNormalEquTerm1)
    println(normalEquTerm2)


  }
}
package ebay

import Helper.InstanceUtilities._
import Helper.String2DoubleUDF
import org.apache.spark.SparkConf
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.{Pipeline, PipelineModel, PipelineStage}
import org.apache.spark.ml.feature.{OneHotEncoder, StandardScaler, StringIndexer, VectorAssembler}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.Row
import org.apache.spark.sql.types._

/**
  * Created by FunctionArlin on 08/2/17.
  */
object ClickPrediction {

  def main(args: Array[String]) = {

    val trainFilePath = "/Users/FunctionArlin/ebay/Task3_test_train/train_data.csv"
    val testFilePath = "/Users/FunctionArlin/ebay/Task3_test_train/test_data.csv"

    val conf = new SparkConf()
      .setAppName("My Logistic Regression")//.set("spark.executor.cores", "5")
      .setMaster("local[*]")

    val sc = initializeSC(conf)
    val sparkSql = initializeSparkSession(conf)

    val dataFields1 = Array[StructField](
      StructField("channel", StringType, nullable = true),
      StructField("client", StringType, nullable = true),
      StructField("position", DoubleType, nullable = true),
      StructField("cpc_in_cent", DoubleType, nullable = true),
      StructField("algorithm", StringType, nullable = true),
      StructField("rankscore", DoubleType, nullable = true)
    )

    val clickedField = Array[StructField](StructField("clicked", DoubleType, nullable = true))

    val dataFields2 = Array[StructField](
      StructField("rankscore2", DoubleType, nullable = true),
      StructField("imps_same_cl_all_ch", DoubleType, nullable = true),
      StructField("ctr_same_cl_same_ch", DoubleType, nullable = true),
      StructField("ctr_diff_cl1_same_ch", DoubleType, nullable = true),
      StructField("ctr_diff_cl2_same_ch", DoubleType, nullable = true),
      StructField("ctr_same_cl_all_ch", DoubleType, nullable = true),
      StructField("seller_ctr_same_cl_same_ch", DoubleType, nullable = true),
      StructField("seller_ctr_diff_cl1_same_ch", DoubleType, nullable = true),
      StructField("seller_ctr_diff_cl2_same_ch", DoubleType, nullable = true),
      StructField("seller_ctr_same_cl_all_ch", DoubleType, nullable = true),
      StructField("ordnung", DoubleType, nullable = true),
      StructField("ctrest", DoubleType, nullable = true),
      StructField("l1", StringType, nullable = true),
      StructField("l2", DoubleType, nullable = true),
      StructField("l1ctr30", DoubleType, nullable = true),
      StructField("l2ctr30", DoubleType, nullable = true),
      StructField("querytokens1", DoubleType, nullable = true),
      StructField("querytokens2", DoubleType, nullable = true),
      StructField("querytokens3+", DoubleType, nullable = true),
      StructField("stopnum", DoubleType, nullable = true),
      StructField("justtitlescore", DoubleType, nullable = true),
      StructField("shinglescore", DoubleType, nullable = true),
      StructField("noshinglescore", DoubleType, nullable = true),
      StructField("noshingleboostscore", DoubleType, nullable = true),
      StructField("noboostscore", DoubleType, nullable = true),
      StructField("titleboostscore", DoubleType, nullable = true),
      StructField("phrasescore", DoubleType, nullable = true),
      StructField("minosprob", DoubleType, nullable = true),
      StructField("simctr", DoubleType, nullable = true)
    )

    val trainDataSchema = StructType(dataFields1 ++ clickedField ++ dataFields2)
    val testDataSchema = StructType(dataFields1 ++ dataFields2)

    val trainDataDF = sparkSql.read.option(key = "header", value = true).schema(trainDataSchema).csv(trainFilePath).na.fill(0.0)
    val testDataDF = sparkSql.read.option(key = "header", value = true).schema(testDataSchema).csv(testFilePath).na.fill(0.0)

    trainDataDF.show()
    trainDataDF.printSchema()

    testDataDF.show()
    testDataDF.printSchema()


    // transform channel to one-hot
    val channelStrIndexer = new StringIndexer().setInputCol("channel").setOutputCol("channelIndexer")
    val channelOneHotEncoder = new OneHotEncoder().setInputCol("channelIndexer").setOutputCol("channelVec")

    // transform client to one-hot
    val clientStrIndexer = new StringIndexer().setInputCol("client").setOutputCol("clientIndexer")
    val clientOneHotEncoder = new OneHotEncoder().setInputCol("clientIndexer").setOutputCol("clientVec")


    // transform algorithm
    val algorithmStrIndexer = new StringIndexer().setInputCol("algorithm").setOutputCol("algorithmIndexer")
    val algorithmOneHotEncoder = new OneHotEncoder().setInputCol("algorithmIndexer").setOutputCol("algorithmVec")


    // transform l1 to dummy.
    val l1StrIndexer = new StringIndexer().setInputCol("l1").setOutputCol("l1Indexer")
    val l1OneHotEncoder = new OneHotEncoder().setInputCol("l1Indexer").setOutputCol("l1Vec")

    // transform l2 to dummy, not feasible, 1686 unique values.
    // Instead, keep it.
//    val l2StrIndexer = new StringIndexer().setInputCol("l2").setOutputCol("l2Indexer")
//    val l2OneHotEncoder = new OneHotEncoder().setInputCol("l2Indexer").setOutputCol("l2Vec")
//    val l2Str2Double = new String2DoubleUDF(_.toDouble).setInputCol("l2").setOutputCol("l2Double")


    // assemble them to features
    val featuresName = "position rankscore rankscore2 imps_same_cl_all_ch ctr_same_cl_same_ch ctr_diff_cl1_same_ch " +
      "ctr_diff_cl2_same_ch ctr_same_cl_all_ch seller_ctr_same_cl_same_ch seller_ctr_diff_cl1_same_ch seller_ctr_diff_cl2_same_ch " +
      "seller_ctr_same_cl_all_ch ordnung ctrest l2 l1ctr30 l2ctr30 querytokens1 querytokens2 querytokens3+ stopnum justtitlescore " +
      "shinglescore noshinglescore noshingleboostscore noboostscore titleboostscore phrasescore minosprob simctr channelVec " +
      "clientVec algorithmVec l1Vec"

    val vecAssembler = new VectorAssembler().setInputCols(featuresName.split(" ")).setOutputCol("features")

    // scale
    val scaler = new StandardScaler().setInputCol("features").setOutputCol("scaledFeatures")

    // logistic regression
    val logisticRegression: LogisticRegression = new LogisticRegression().setLabelCol("clicked")
      .setFeaturesCol("scaledFeatures").setMaxIter(100).setFitIntercept(true).setWeightCol("cpc_in_cent")

    val stages = Array[PipelineStage](channelStrIndexer, channelOneHotEncoder, clientStrIndexer, clientOneHotEncoder,
      algorithmStrIndexer, algorithmOneHotEncoder, l1StrIndexer, l1OneHotEncoder, vecAssembler, scaler, logisticRegression)

    val pipeline = new Pipeline().setStages(stages)

    val pipelineModel: PipelineModel = pipeline.fit(trainDataDF)
    val trTrainDataDF = pipelineModel.transform(trainDataDF)

    trTrainDataDF.show()
    trTrainDataDF.printSchema()

    val logisticRegModel: LogisticRegressionModel = pipelineModel.stages(stages.length - 1).asInstanceOf[LogisticRegressionModel]

    val theta: Vector = logisticRegModel.coefficients
    val intercept: Double = logisticRegModel.intercept
    println("ML Theta: " + theta.toArray.mkString(", "))
    println("ML Intercept: " + intercept)
    println("ML Objective history: " + logisticRegModel.summary.objectiveHistory.mkString(", "))


    trTrainDataDF.select(logisticRegModel.getProbabilityCol, logisticRegModel.getPredictionCol).take(20).foreach(println)

    val biMetrics = new MulticlassMetrics(trTrainDataDF.select(logisticRegModel.getPredictionCol, "clicked").rdd.map{
      case Row(prediction: Double, label: Double) => (prediction, label)
    })

    println("ML accuracy: " + biMetrics.accuracy)

  }

}

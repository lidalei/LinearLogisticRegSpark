package leafclassification

import Helper.InstanceUtilities.initializeSparkSession
import Helper.{MultipleClassificationCrossEntropyEvaluator, UDFStringIndexer}
import org.apache.spark.SparkConf
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.feature.StandardScaler
//import org.apache.spark.ml.attribute.Attribute
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.feature.{StringIndexerModel, VectorAssembler}
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.sql.types._

import scala.collection.mutable.HashMap

/**
  * Created by Sophie on 2/10/17.
  */
object LeafClassification {

  val DATA_FOLDER = "/Users/Sophie/Downloads/LeafClassificationData/"

  def main(args: Array[String]): Unit = {

    val sparkConf: SparkConf = new SparkConf().setAppName("Kaggle Leaf Classification").setMaster("local[*]")

    val sparkSql = initializeSparkSession(sparkConf)

    val sc = sparkSql.sparkContext

    sc.setLogLevel("WARN")

    // read training and test data
    val trainFilePath = DATA_FOLDER + "train.csv"
    val testFilePath = DATA_FOLDER + "test.csv"

    val sampleSubmissionFilePath = DATA_FOLDER + "sample_submission.csv"

    val idField = StructField("id", IntegerType, nullable = false)
    val speciesField = StructField("species", StringType, nullable = false)

    val marginsNames = (1 to 64).toArray.map("margin" + _)
    val marginsFields = marginsNames.map(StructField(_, DoubleType, nullable = true))

    val shapesNames = (1 to 64).toArray.map("shape" + _)
    val shapesFields = shapesNames.map(StructField(_, DoubleType, nullable = true))

    val texturesNames = (1 to 64).toArray.map("texture" + _)
    val textureFields = texturesNames.map(StructField(_, DoubleType, nullable = true))

    val trainDataSchema: StructType = StructType(Array[StructField](idField, speciesField)
      ++ marginsFields ++ shapesFields ++ textureFields)

    val testDataSchema: StructType = StructType(Array[StructField](idField)
      ++ marginsFields ++ shapesFields ++ textureFields)

    val trainDF = sparkSql.read.option(key = "header", value = true).schema(trainDataSchema).csv(trainFilePath)
    val testDF = sparkSql.read.option(key = "header", value = true).schema(testDataSchema).csv(testFilePath)

//    println("Training data:")
//    trainDF.show()
//    trainDF.printSchema()

//    println("Test data:")
//    testDF.show()
//    testDF.printSchema()


    // submission indexer
    val species = sc.textFile(sampleSubmissionFilePath).first().split(",").tail

    // transform species to label
    val speciesStrIndexer = new UDFStringIndexer().setInputCol(speciesField.name).setOutputCol("label").setLabels(species)

    // assemble all margin, shape and texture features
    val featuresVecAssembler = new VectorAssembler()
      .setInputCols(marginsNames ++ shapesNames ++ texturesNames).setOutputCol("rawFeatures")

    val scaler = new StandardScaler().setInputCol("rawFeatures").setOutputCol("features").setWithMean(true).setWithStd(true)

    // logistic regression model
    val logReg = new LogisticRegression().setFeaturesCol("features").setLabelCol("label").setFitIntercept(true).setMaxIter(400)

//    val rndForest = new RandomForestClassifier().setFeaturesCol("features").setLabelCol("label").setMaxMemoryInMB(500)

    // build pipeline
    val stages = Array[PipelineStage](speciesStrIndexer, featuresVecAssembler, scaler, logReg)
    val pipeline = new Pipeline().setStages(stages)

    // hyper-parameter tuning through cross validation
//    val evaluator = new MulticlassClassificationEvaluator()
//      .setLabelCol(logReg.getLabelCol).setPredictionCol(logReg.getPredictionCol).setMetricName("accuracy")

    val evaluator = new MultipleClassificationCrossEntropyEvaluator().setLabelCol(logReg.getLabelCol)
      .setProbabilityCol(logReg.getProbabilityCol)

    val paramGrid = new ParamGridBuilder()
//        .addGrid(rndForest.impurity, Array("entropy", "gini"))
//        .addGrid(rndForest.maxDepth, Array(3, 5, 9))
//        .addGrid(rndForest.minInstancesPerNode, Array(1, 3, 9))
//        .addGrid(rndForest.numTrees, Array(5, 10, 20))
      .addGrid(logReg.regParam, Array(0.000003, 0.00003, 0.0003, 0.003, 0.03, 0.3))
//      .addGrid(logReg.elasticNetParam, Array(0.4, 0.5, 0.8))
      .build()


    val cv = new CrossValidator().setEstimator(pipeline).setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid).setNumFolds(10)

    val cvModel: CrossValidatorModel = cv.fit(trainDF)
    println("Average cross entropy: " + cvModel.avgMetrics.mkString(","))

    val pipelineModel= cvModel.bestModel.asInstanceOf[PipelineModel]
    val logRegModel = pipelineModel.stages(stages.length - 1).asInstanceOf[LogisticRegressionModel]

//    val rndForestModel = pipelineModel.stages(stages.length - 1).asInstanceOf[RandomForestClassificationModel]

//    println("Logistic regression theta: " + logRegModel.coefficientMatrix)
//    println("Logistic regression intercept: " + logRegModel.interceptVector)
    println(
//      "Best parameters, maxDepth: " + rndForestModel.getMaxDepth + ", minInstancesPerNode: " +
//      rndForestModel.getMinInstancesPerNode + ", numTrees: " + rndForestModel.getNumTrees +
      "logReg reg: " + logRegModel.getRegParam)

//    if(logRegModel.hasSummary) {
//      println("Logistic regression object history: " + logRegModel.summary.objectiveHistory)
//    }

    // make predictions. StringIndexer would skip the species column, see source code.
    val predictions = pipelineModel.transform(testDF).select(idField.name, logRegModel.getProbabilityCol)
    predictions.show()


    // write to json file to be processed in Python
    predictions.write.json(DATA_FOLDER + "submission")

    // used indexer
    val speciesStrIndexerModel = pipelineModel.stages(0).asInstanceOf[StringIndexerModel]
    val species2LabelMap = HashMap[String, Int]()
    speciesStrIndexerModel.labels.zipWithIndex.map(e => species2LabelMap.put(e._1, e._2.toInt))
    println("Species string 2 index map: " + species2LabelMap)

//    println("Species string 2 index " + Attribute.fromStructField(predictions.schema(bestModel.getPredictionCol)).toString)

    val targetSpecies2LabelMap = HashMap[String, Int]()
    species.zipWithIndex.map(e => targetSpecies2LabelMap.put(e._1, e._2))
    println("Target species 2 label map: " + targetSpecies2LabelMap)

    sparkSql.stop()
  }

}

package leafclassification

import Helper.InstanceUtilities.{initializeSC, initializeSparkSession}
import Helper.String2DoubleUDF
import org.apache.spark.SparkConf
import org.apache.spark.ml.attribute.Attribute
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{StringIndexer, StringIndexerModel, VectorAssembler}
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.sql.types._

import scala.collection.mutable

/**
  * Created by Sophie on 2/10/17.
  */
object LeafClassification {

  val DATA_FOLDER = "/Users/Sophie/Downloads/LeafClassificationData/"

  def main(args: Array[String]) = {

    val sparkConf: SparkConf = new SparkConf().setAppName("Kaggle Leaf Classification").setMaster("local[*]")

    val sc = initializeSC(sparkConf)
    val sparkSql = initializeSparkSession(sparkConf)

    sc.setLogLevel("WARN")

//    import sparkSql.implicits._

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

    println("Training data:")
    trainDF.show()
//    trainDF.printSchema()

    println("Test data:")
    testDF.show()
//    testDF.printSchema()

    val species = sc.textFile(sampleSubmissionFilePath).first().split(",").tail
    val species2LabelMap = new mutable.HashMap[String, Double]()
    species.zipWithIndex.map(e => species2LabelMap.put(e._1, e._2.toDouble))

    // transform species to label
    val speciesStr2Double = new String2DoubleUDF(species2LabelMap.getOrElse(_, 0.0))
      .setInputCol(speciesField.name).setOutputCol("label")
    val speciesStrIndexer = new StringIndexer().setInputCol(speciesField.name).setOutputCol("label")

    // assemble all margin, shape and texture features
    val featuresVecAssembler = new VectorAssembler()
      .setInputCols(marginsNames ++ shapesNames ++ texturesNames).setOutputCol("features")

    // logistic regression model
    val logReg = new LogisticRegression().setFeaturesCol("features").setLabelCol("label")
      .setFitIntercept(true).setMaxIter(100)

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label").setPredictionCol(logReg.getPredictionCol).setMetricName("accuracy")

    val paramGrid = new ParamGridBuilder().addGrid(logReg.regParam, Array(0.001, 0.01))
//      .addGrid(logReg.elasticNetParam, Array(0.4, 0.5, 0.8))
      .build()

    val cv = new CrossValidator().setEstimator(logReg).setEvaluator(evaluator).setEstimatorParamMaps(paramGrid).setNumFolds(3)

    // build pipeline
    val stages = Array[PipelineStage](speciesStr2Double, featuresVecAssembler, cv)

    val pipeline = new Pipeline().setStages(stages)

    val pipelineModel = pipeline.fit(trainDF)

    val cvModel = pipelineModel.stages(stages.length - 1).asInstanceOf[CrossValidatorModel]

    println("Average accuracy: " + cvModel.avgMetrics.mkString(","))

    val bestModel = cvModel.bestModel.asInstanceOf[LogisticRegressionModel]

    println("Logistic regression theta: " + bestModel.coefficientMatrix)

    println("Logistic regression intercept: " + bestModel.interceptVector)

    if(bestModel.hasSummary) {
      println("Logistic regression object history: " + bestModel.summary.objectiveHistory)
    }

    // make predictions. StringIndexer would skip the species column, see source code.
    val predictions = pipelineModel.transform(testDF)//.select(idField.name, bestModel.getPredictionCol, bestModel.getProbabilityCol)

    predictions.show()

    println("Species string 2 index " + Attribute.fromStructField(predictions.schema(bestModel.getPredictionCol)).toString)

//    predictions.write.csv(DATA_FOLDER + "predictions")

  }



}

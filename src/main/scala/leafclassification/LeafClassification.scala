package leafclassification

import Helper.InstanceUtilities.initializeSparkSession
import Helper.UDFStringIndexer
import org.apache.spark.SparkConf
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.feature.PolynomialExpansion
//import org.apache.spark.ml.attribute.Attribute
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{StringIndexerModel, VectorAssembler}
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.sql.types._

import scala.collection.mutable.HashMap

/**
  * Created by Sophie on 2/10/17.
  */
object LeafClassification {

  val DATA_FOLDER = "/Users/Sophie/Downloads/LeafClassificationData/"

  def main(args: Array[String]) = {

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
      .setInputCols(marginsNames ++ shapesNames ++ texturesNames).setOutputCol("features")

    // polynomial expansion
    val polyExpansion = new PolynomialExpansion().setInputCol("features").setOutputCol("polyFeatures")

    // logistic regression model
    val logReg = new LogisticRegression().setFeaturesCol("features").setLabelCol("label")
      .setFitIntercept(true).setMaxIter(100)

    // build pipeline
    val stages = Array[PipelineStage](speciesStrIndexer, featuresVecAssembler, polyExpansion, logReg)
    val pipeline = new Pipeline().setStages(stages)

    // hyper-parameter tuning through cross validation
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label").setPredictionCol(logReg.getPredictionCol).setMetricName("accuracy")

    val paramGrid = new ParamGridBuilder().addGrid(logReg.regParam, Array(0.0, 0.001, 0.01))
//      .addGrid(logReg.elasticNetParam, Array(0.4, 0.5, 0.8))
        .addGrid(polyExpansion.degree, Array(1))
      .build()

    val cv = new CrossValidator().setEstimator(pipeline).setEvaluator(evaluator).setEstimatorParamMaps(paramGrid).setNumFolds(3)
    val cvModel: CrossValidatorModel = cv.fit(trainDF)
    println("Average accuracy: " + cvModel.avgMetrics.mkString(","))

    val pipelineModel= cvModel.bestModel.asInstanceOf[PipelineModel]
    val logRegModel = pipelineModel.stages(stages.length - 1).asInstanceOf[LogisticRegressionModel]

//    println("Logistic regression theta: " + logRegModel.coefficientMatrix)
//    println("Logistic regression intercept: " + logRegModel.interceptVector)
    println("Best parameters, reg: " + logRegModel.getRegParam +", poly degree: "
      + pipelineModel.stages(2).asInstanceOf[PolynomialExpansion].getDegree)

    if(logRegModel.hasSummary) {
      println("Logistic regression object history: " + logRegModel.summary.objectiveHistory)
    }

    // make predictions. StringIndexer would skip the species column, see source code.
    val predictions = pipelineModel.transform(testDF).select(idField.name, logRegModel.getProbabilityCol)
    predictions.show()


    // write to json file to be processed in Python
    predictions.write.json(DATA_FOLDER + "submission")

    // used indexer
    val speciesStrIndexerModel = pipelineModel.stages(0).asInstanceOf[StringIndexerModel]
    val species2LabelMap = new HashMap[String, Int]()
    speciesStrIndexerModel.labels.zipWithIndex.map(e => species2LabelMap.put(e._1, e._2.toInt))
    println("Species string 2 index map: " + species2LabelMap)

//    println("Species string 2 index " + Attribute.fromStructField(predictions.schema(bestModel.getPredictionCol)).toString)

    val targetSpecies2LabelMap = new HashMap[String, Int]()
    species.zipWithIndex.map(e => targetSpecies2LabelMap.put(e._1, e._2))
    println("Target species 2 label map: " + targetSpecies2LabelMap)

    sparkSql.stop()
  }

}

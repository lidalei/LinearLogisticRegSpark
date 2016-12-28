package Helper

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.util.LongAccumulator

/**
  * Created by Sophie on 12/2/16.
  * This object is used to hold Instance, InstanceWithPredition and InstanceWithPredictionProb case classes.
  * Besides, it holds the Confusion Matrix case class.
  */

object InstanceUtilities {

  def initializeSC(conf: SparkConf): SparkContext = {
    SparkContext.getOrCreate(conf)
  }

  def initializeSparkSession(conf: SparkConf): SparkSession = {
    SparkSession.builder().config(conf).getOrCreate()
  }


  case class Instance(label: Double, features: Vector)

  /**
    * The case class to hold Instance and its predictionProb to be class 1
    * @param label: true class
    * @param features: features used to predict
    * @param predictionProb: probability to be class 1
    */
  case class InstanceWithPredictionProb(label: Double, features: Vector, predictionProb: Double)
  case class InstanceWithPrediction(label: Double, features: Vector, prediction: Double)

  case class ConfusionMatrix(truePositiveAcc: LongAccumulator, trueNegativeAcc: LongAccumulator, falsePositiveAcc: LongAccumulator, falseNegativeAcc: LongAccumulator) {
    override def toString: String = {
      "truePositive: " + truePositiveAcc.value + ", trueNegative: " + trueNegativeAcc.value + ", falsePositive: " + falsePositiveAcc.value + ", falseNegative: " + falseNegativeAcc.value
    }
  }

  def score(confusionMatrix: ConfusionMatrix, scoreType: String = "accuracy"): Double = {
    confusionMatrix match {
      case ConfusionMatrix(truePositiveAcc, trueNegativeAcc, falsePositiveAcc, falseNegativeAcc) => {

        val predictedPositive = truePositiveAcc.value + falsePositiveAcc.value
        val predictedNegative = trueNegativeAcc.value + falseNegativeAcc.value

        val actualPositive = truePositiveAcc.value + falseNegativeAcc.value
        val actualNegative = trueNegativeAcc.value + falsePositiveAcc.value

        val correctPredictions = truePositiveAcc.value + trueNegativeAcc.value

        scoreType match {
          case "accuracy" => correctPredictions.toDouble / (predictedNegative + predictedPositive)
          case "precision" => truePositiveAcc.value.toDouble / predictedPositive
          case "recall" => truePositiveAcc.value.toDouble / actualPositive
        }
      }
    }
  }


  def df2RDD(df: DataFrame, featuresVecName: String, label: String, addAllOneCol: Boolean = false): RDD[Instance] = {
    if (addAllOneCol) {
      df.select(col(label), col(featuresVecName)).rdd.map({
        // append all-one column in the end of features vector
        case Row(label: Double, features: Vector) => Instance(label, Vectors.dense(features.toArray ++ Array(1.0)))
      })
    }
    else {
      df.select(col(label), col(featuresVecName)).rdd.map({
        case Row(label: Double, features: Vector) => Instance(label, features)
      })
    }

  }


}
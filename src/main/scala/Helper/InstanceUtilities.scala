package Helper

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

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


  /**
    *
    * @param label: true class
    * @param features: features used to predict
    */
  case class Instance(label: Double, features: Vector)

  /**
    * The case class to hold Instance and its predictionProb to be class 1
    * @param label: true class
    * @param predictionProb: probability to be class 1
    * @param prediction: predicted label
    */
  case class InstanceWithPrediction(label: Double, predictionProb: Double, prediction: Double)

  // for regression
  case class InstanceWithPredictionReg(label: Double, prediction: Double)

  case class ConfusionMatrix(truePositive: Long, trueNegative: Long, falsePositive: Long, falseNegative: Long) {
    override def toString: String = {
      "truePositive: " + truePositive + ", trueNegative: " + trueNegative + ", falsePositive: " + falsePositive + ", falseNegative: " + falseNegative
    }
  }

  def score(confusionMatrix: ConfusionMatrix, scoreType: String = "accuracy"): Double = {
    confusionMatrix match {
      case ConfusionMatrix(truePositive, trueNegative, falsePositive, falseNegative) => {

        val predictedPositive = truePositive + falsePositive
        val predictedNegative = trueNegative + falseNegative

        val actualPositive = truePositive + falseNegative
        val actualNegative = trueNegative + falsePositive

        val correctPredictions = truePositive + trueNegative

        scoreType match {
          case "accuracy" => correctPredictions.toDouble / (predictedNegative + predictedPositive)
          case "precision" => truePositive.toDouble / predictedPositive
          case "recall" => truePositive.toDouble / actualPositive
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
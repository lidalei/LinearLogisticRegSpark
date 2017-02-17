package Helper

import org.apache.spark.ml.evaluation.Evaluator
import org.apache.spark.ml.param.{Param, ParamMap, Params}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{Dataset, Row}
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.ml.linalg.Vector

/**
  * Created by Sophie on 2/17/17.
  */
class MultipleClassificationCrossEntropyEvaluator(val uid: String) extends Evaluator with Params {

  def this() = this(Identifiable.randomUID("multiClassXEntropy"))

  // The probability column used to compute cross entropy. It should be Vector type.
  final val probabilityCol: Param[String] = new Param[String](this, "probabilityCol", "prediction probability column")

  def setProbabilityCol(value: String): this.type = set(probabilityCol, value)
  setDefault(probabilityCol, "probability")

  def getProbabilityCol(): String = $(probabilityCol)

  // The true label column. It should be Double- or Integer- type.
  final val labelCol: Param[String] = new Param[String](this, "labelCol", "true label column")

  def setLabelCol(value: String): this.type = set(labelCol, value)
  setDefault(labelCol, "label")

  def getLabelCol(): String = $(labelCol)

  override def evaluate(dataset: Dataset[_]): Double = {
    val fieldNames = dataset.schema.fieldNames

    if(!fieldNames.contains($(probabilityCol)) || !fieldNames.contains($(labelCol))) {
      throw new NoSuchFieldException(s"No $$(probabilityCol) or $$(labelCol) column.")
    }

    val probabilityAndLabels =
      dataset.select(col($(probabilityCol)), col($(labelCol)).cast(IntegerType)).rdd.map {
        case Row(probability: Vector, label: Int) => (probability, label)
      }

    val crossEntropy: Double = - probabilityAndLabels.map{
      case (probability: Vector, label: Int) => math.log(math.max(math.min(1 - 1e-15, probability(label)), 1e-15))
    }.mean()

    crossEntropy
  }

  override def isLargerBetter: Boolean = false

  def copy(extra: ParamMap): MultipleClassificationCrossEntropyEvaluator = defaultCopy(extra)
}

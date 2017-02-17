package Helper

import org.apache.spark.ml.feature.{StringIndexer, StringIndexerModel}
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.Dataset

/**
  * Created by Sophie on 2/11/17.
  */
class UDFStringIndexer(uid: String) extends StringIndexer(uid) {
  def this() = this(Identifiable.randomUID("udfStrIdx"))

  // Treat labels as a parameter. It specifies the labels mapped to indices starting from 0.
  // For example, Array[String]("cat", "dog", "mouse", ...) maps "cat" to 0, "dog" to 1 and "mouse" to 2, etc.
  final val labels: Param[Array[String]] = new Param[Array[String]](this, "labels", "user specified labels")

  // Default value is an empty Array. It will invoke builtin fit function.
  setDefault(labels, Array[String]())

  def setLabels(udfLabels: Array[String]): this.type = set(labels, udfLabels)

  final def getLabels(): Array[String] = $(labels)

  override def fit(dataset: Dataset[_]): StringIndexerModel = {
    if($(labels).isEmpty) {
      super.fit(dataset)
    }
    else {
      transformSchema(dataset.schema, logging = true)
      copyValues(new StringIndexerModel($(labels)).setParent(this))
    }
  }
}

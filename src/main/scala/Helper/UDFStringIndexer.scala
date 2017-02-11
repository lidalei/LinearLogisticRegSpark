package Helper

import org.apache.spark.ml.feature.{StringIndexer, StringIndexerModel}
import org.apache.spark.sql.Dataset

/**
  * Created by Sophie on 2/11/17.
  */
class UDFStringIndexer(labels: Array[String]) extends StringIndexer {

  override def fit(dataset: Dataset[_]): StringIndexerModel = {
    transformSchema(dataset.schema, logging = true)
    copyValues(new StringIndexerModel(uid, labels).setParent(this))
  }
}

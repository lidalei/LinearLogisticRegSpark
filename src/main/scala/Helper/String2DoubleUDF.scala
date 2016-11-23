package Helper

/**
  * Created by Sophie on 11/23/16.
  */

import org.apache.spark.ml.UnaryTransformer
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.ml.param.ParamMap

import scala.util.{Failure, Success, Try}

class String2DoubleUDF(override val uid: String, val udf: String => Double)
  extends UnaryTransformer[String, Double, String2DoubleUDF] {

  def this(udf: String => Double) = this(Identifiable.randomUID("string2DoubleUDF"), udf)

  // This is super important to deal with "NA" in test data
  override protected def createTransformFunc: String => Double = {
    def trFun(str: String): Double = {
      Try[Double](udf(str)) match {
        case Success(x) => x
        case Failure(e) => 0.0
      }
    }
    trFun
  }

  override protected def outputDataType: DoubleType = {
    DoubleType
  }

  override def copy(extra: ParamMap): String2DoubleUDF = {
    new String2DoubleUDF(udf).setInputCol(getInputCol).setOutputCol(getOutputCol)
  }

}

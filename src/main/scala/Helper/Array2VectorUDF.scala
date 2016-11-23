package Helper

/**
  * Created by Sophie on 11/23/16.
  */

import org.apache.spark.ml.UnaryTransformer
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.hack.VectorType
import org.apache.spark.ml.param.ParamMap

import scala.collection.mutable.WrappedArray

class Array2VectorUDF(override val uid: String, val udf: String => Double)
  extends UnaryTransformer[WrappedArray[String], Vector, Array2VectorUDF] {

  def this(udf: String => Double) = this(Identifiable.randomUID("arrayToVector"), udf)

  // udf applied on each element
  override protected def createTransformFunc: WrappedArray[String] => Vector = (strArr: WrappedArray[String]) => strArr match {
    case (x: WrappedArray[String]) => {Vectors.dense(x.array.map { udf })}
  }

  override protected def outputDataType: VectorType = {
    new VectorType
  }

  override def copy(extra: ParamMap): Array2VectorUDF = {
    new Array2VectorUDF(udf).setInputCol(getInputCol).setOutputCol(getOutputCol)
  }
}

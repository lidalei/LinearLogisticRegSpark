package Helper

/**
  * Created by Sophie on 11/23/16.
  */

import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.UnaryTransformer
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.ml.param.ParamMap

class Vector2DoubleUDF(override val uid: String, val udf: Vector => Double)
  extends UnaryTransformer[Vector, Double, Vector2DoubleUDF] {

  // udf applied on the vector
  def this(udf: Vector => Double) = this(Identifiable.randomUID("vector2DoubleUDF"), udf)

  override protected def createTransformFunc: Vector => Double = udf

  override protected def outputDataType: DoubleType = {
    DoubleType
  }

  override def copy(extra: ParamMap): Vector2DoubleUDF = {
    new Vector2DoubleUDF(udf).setInputCol(getInputCol).setOutputCol(getOutputCol)
  }
}

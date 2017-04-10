package Helper

/**
  * Created by Sophie on 2/12/17.
  */

import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.UnaryTransformer
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.types.{ArrayType, DoubleType}

class Vector2ArrayUDF(override val uid: String, val udf: Vector => Array[Double])
  extends UnaryTransformer[Vector, Array[Double], Vector2ArrayUDF] {

  // udf applied on the vector
  def this(udf: Vector => Array[Double]) = this(Identifiable.randomUID("vector2ArrayUDF"), udf)

  override protected def createTransformFunc: Vector => Array[Double] = udf

  override protected def outputDataType: ArrayType = {
    ArrayType(DoubleType, false)
  }

  override def copy(extra: ParamMap): Vector2ArrayUDF = {
    new Vector2ArrayUDF(udf).setInputCol(getInputCol).setOutputCol(getOutputCol)
  }
}

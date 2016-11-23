package Helper

import org.apache.spark.ml.linalg.{Matrices, Vector, Vectors}
import VectorManipulation._
import org.scalatest.FunSuite
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class VectorManipulationTest extends FunSuite {

  trait TestVectorManipulation {
    val v1:Vector = Vectors.dense(1.0, 2.0, 3.0)
    val v2:Vector = Vectors.dense(2.0, 3.0, 4.0)
  }

  test("test VectorManipulation functions") {
    new TestVectorManipulation {
      assert(vecInnerProduct(v1, v2) === 20.0)

      assert(outerVecProduct(v1, v2) === Matrices.dense(v1.size, v2.size, Array(2.0, 4.0, 6.0, 3.0, 6.0, 9.0, 4.0, 8.0, 12.0)))

      assert(vecNegative(v1) === Vectors.dense(Array(-1.0, -2.0, -3.0)))

      assert(vecAdd(v1, v2) === Vectors.dense(Array(3.0, 5.0, 7.0)))

    }
  }

}

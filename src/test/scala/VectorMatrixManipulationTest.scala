package Helper

import org.apache.spark.ml.linalg.{DenseMatrix, Matrices, Matrix, Vector, Vectors}
import VectorMatrixManipulation._
import org.scalatest.FunSuite
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class VectorMatrixManipulationTest extends FunSuite {

  trait TestVectorManipulation {
    val v1:Vector = Vectors.dense(1.0, 2.0, 3.0)
    val v2:Vector = Vectors.dense(2.0, 3.0, 4.0)

    val m1: DenseMatrix = new DenseMatrix(3, 3, Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0))
    val m2: DenseMatrix = new DenseMatrix(3, 3, Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0))

    val m3: DenseMatrix = new DenseMatrix(3, 3, Array(2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0))
  }

  test("test VectorManipulation functions") {
    new TestVectorManipulation {
      assert(vecInnerProduct(v1, v2) === 20.0)

      assert(outerVecProduct(v1, v2) === Matrices.dense(v1.size, v2.size, Array(2.0, 4.0, 6.0, 3.0, 6.0, 9.0, 4.0, 8.0, 12.0)))

      assert(vecNegative(v1) === Vectors.dense(Array(-1.0, -2.0, -3.0)))

      assert(vecAdd(v1, v2) === Vectors.dense(Array(3.0, 5.0, 7.0)))

      assert(vecScale(v1, 4) === Vectors.dense(Array(4.0, 8.0, 12.0)))

      assert(matrixAdd(m1, m2) === m3)

      assert(matrixScale(m1, 2.0) === m3)

      assert(outerVecProduct(v1) === Vectors.dense(Array(1, 2, 4, 3, 6, 9.0)))

      assert(upperTriangle(4, 1.0) === Vectors.dense(Array(1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)))

//      assert(randVec(5) === Vectors.dense(Array(1.0)))

    }
  }

}

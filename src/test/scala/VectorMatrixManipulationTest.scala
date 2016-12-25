package Helper

import Helper.InstanceUtilities.{ConfusionMatrix, initializeSC, score}
import org.apache.spark.ml.linalg.{DenseMatrix, Matrices, Matrix, Vector, Vectors}
import VectorMatrixManipulation._
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.util.LongAccumulator
import org.scalatest.FunSuite
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class VectorMatrixManipulationTest extends FunSuite {

  trait TestVectorManipulation {
    val v1:Vector = Vectors.dense(1.0, 2.0, 3.0)
    val v2:Vector = Vectors.dense(2.0, 3.0, 4.0)

    val v3:Vector = Vectors.dense(2.0, 3.0, -4.0)

    val m1: DenseMatrix = new DenseMatrix(3, 3, Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0))
    val m2: DenseMatrix = new DenseMatrix(3, 3, Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0))

    val m3: DenseMatrix = new DenseMatrix(3, 3, Array(2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0))

    val sc: SparkContext = initializeSC(conf = new SparkConf().setAppName("Test").setMaster("local"))

    // accumulators to store confusion matrix
    val truePositiveAcc: LongAccumulator = sc.longAccumulator("truePositive")
    val falsePositiveAcc: LongAccumulator = sc.longAccumulator("falsePositive")
    val trueNegativeAcc: LongAccumulator = sc.longAccumulator("trueNegative")
    val falseNegativeAcc: LongAccumulator = sc.longAccumulator("falseNegative")

    truePositiveAcc.add(100)

    trueNegativeAcc.add(50)

    falsePositiveAcc.add(10)

    falseNegativeAcc.add(5)

    val confusionMatrix: ConfusionMatrix = ConfusionMatrix(truePositiveAcc, trueNegativeAcc, falsePositiveAcc, falseNegativeAcc)
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

      println(m1.toString())

      assert(matrixMultiplyVec(Vectors.dense(m1.toArray), v1) === Vectors.dense(Array(30.0, 36.0, 42.0)))


      assert(score(confusionMatrix, "accuracy") === 150.0 / 165)

      assert(score(confusionMatrix, "precision") === 100.0 / 110)

      assert(score(confusionMatrix, "recall") === 100.0 / 105)

      assert(vecNormPower(v1, 2) === 14)

      assert(vecNormPower(v2, 1) === 9)

      assert(vecNormPower(v3, 1) === 9)

    }
  }

}

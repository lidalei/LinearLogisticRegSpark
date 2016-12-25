package Helper

/**
  * Created by Sophie on 11/23/16.
  */

import com.github.fommil.netlib.LAPACK.{getInstance => lapack}
import org.apache.spark.ml.linalg.Matrices
import org.netlib.util.intW
//import org.apache.spark.SparkContext
import org.apache.spark.ml.linalg.{DenseMatrix, Matrix, Vector, Vectors}
//import org.apache.spark.rdd.RDD

object VectorMatrixManipulation {


  /**
    * Generate a random Vector of a specific size
    * @param n
    * @return
    */
  def randVec(n: Int): Vector = {
    Vectors.dense((0 to n - 1).toArray.map(_ => math.random))
  }


  def vecNormPower(v: Vector, order: Int): Double = {
    v.toArray.map(e => math.pow(math.abs(e), order)).sum
  }

  /**
    * Compute outer product of v with itself
    * return the upper triangle with column as the order of serialization
    * @param v
    * @return
    */
  def outerVecProduct(v: Vector): Vector = {
    Vectors.dense((0 until v.size).flatMap((i: Int) => (0 to i).map((j: Int) => v(i) * v(j))).toArray)

//    for{
//      i <- 0 until v.size
//      j <- 0 to i
//    } yield v(i) * v(j)

  }

  /**
    * generate the vector form of an upper triangle identity matrix
    * @param n
    * @param diagValue
    * @return
    */
  def upperTriangle(n: Int, diagValue: Double): Vector = {
    Vectors.dense((0 until n).flatMap((i: Int) => (0 to i).map((j: Int) => if(i == j) diagValue else 0.0)).toArray)
  }


  /**
    * outer product of two vectors
    * @param v1, column vector
    * @param v2, row vector
    * @return
    */
  def outerVecProduct(v1: Vector, v2: Vector): DenseMatrix = {
    val it1 = 0 until v1.size
    val it2 = 0 until v2.size
    // the result is in the same format with numpy
    new DenseMatrix(v1.size, v2.size, it2.flatMap(i => it1.map(j => v1(j) * v2(i))).toArray)
  }

  /**
    * addition of two vectors
    * @param v1
    * @param v2
    * @return
    */
  def vecAdd(v1: Vector, v2: Vector): Vector = {
    Vectors.dense((0 until v1.size).map(i => v1(i) + v2(i)).toArray)
  }

  /**
    * negative of a vector
    * @param v1
    * @return
    */
  def vecNegative(v1: Vector): Vector = {
    Vectors.dense(v1.toArray.map(-_))
  }

  /**
    * inner product of two vectors
    * @param v1
    * @param v2
    * @return
    */
  def vecInnerProduct(v1: Vector, v2: Vector): Double = {
    (0 until v1.size).map(i => v1(i) * v2(i)).sum
  }


  /**
    * scale a vector
    * @param v1
    * @param s
    * @return
    */
  def vecScale(v1: Vector, s: Double): Vector = {
    Vectors.dense(v1.toArray.map(_ * s))
  }


  def matrixAdd(m1: DenseMatrix, m2: DenseMatrix): DenseMatrix = {
    val m1Arr = m1.toArray
    val m2Arr = m2.toArray
    new DenseMatrix(m1.numRows, m1.numCols, m1Arr.indices.map(i => m1Arr(i) + m2Arr(i)).toArray)
  }

  def matrixScale(m1: Matrix, s: Double): DenseMatrix = {
    new DenseMatrix(m1.numRows, m1.numCols, m1.toArray.map(_ * s))
  }

//
//  def matrix2RDD(sc: SparkContext, m1: DenseMatrix): RDD[Vector] = {
//
//    sc.parallelize((0 until m1.numRows).map((i: Int) => Vectors.dense((0 until m1.numCols).map((j: Int) => m1(j, i)).toArray)))
//
//  }

  /**
    *
    * @param A, column order matrix
    * @param v
    * @return
    */
  def matrixMultiplyVec(A: Vector, v: Vector): Vector = {
    val dimension = v.size
    Matrices.dense(dimension, dimension, A.toArray).multiply(v)
  }


  /**
    * Compute Cholesky decomposition.
    * source code from org.apache.spark.mllib.linalg.CholeskyDecomposition
    */

  /**
    * Solves a symmetric positive definite linear system via Cholesky factorization.
    * The input arguments are modified in-place to store the factorization and the solution.
    * @param A the upper triangular part of A
    * @param bx right-hand side, with only one column, see dppsv third parameter
    * @return the solution array
    */
  def solve(A: Array[Double], bx: Array[Double]): Array[Double] = {
    val k = bx.length
    val info = new intW(0)
    lapack.dppsv("U", k, 1, A, bx, k, info)
    val code = info.`val`
    assert(code == 0, s"lapack.dppsv returned $code.")
    bx
  }


  // A * x = B
  def solveLinearEquations(AP: Array[Double], B: Array[Double], N: Int, NRHS: Int): Array[Double] = {
    val info = new intW(0)
    lapack.dppsv("U", N, NRHS, AP, B, N, info)
    val code = info.`val`
    assert(code == 0, s"lapack.dppsv returned $code.")
    B
  }


  /**
    * Computes the inverse of a real symmetric positive definite matrix A
    * using the Cholesky factorization A = U**T*U.
    * The input arguments are modified in-place to store the inverse matrix.
    * @param UAi the upper triangular factor U from the Cholesky factorization A = U**T*U
    * @param k the dimension of A
    * @return the upper triangle of the (symmetric) inverse of A
    */
  def inverse(UAi: Array[Double], k: Int): Array[Double] = {
    val info = new intW(0)
    lapack.dpptri("U", k, UAi, info)
    val code = info.`val`
    assert(code == 0, s"lapack.dpptri returned $code.")
    UAi
  }

}

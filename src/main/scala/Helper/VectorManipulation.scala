package Helper

/**
  * Created by Sophie on 11/23/16.
  */

import org.apache.spark.ml.linalg.{Vector, Vectors, Matrix, Matrices}

object VectorManipulation {

  /**
    * outer product of two vectors
    * @param v1, column vector
    * @param v2, row vector
    * @return
    */
  def outerVecProduct(v1: Vector, v2: Vector): Matrix = {
    val it1 = 0 until v1.size
    val it2 = 0 until v2.size
    // the result is in the same format with numpy
    Matrices.dense(v1.size, v2.size, it2.flatMap(i => it1.map(j => v1(j) * v2(i))).toArray)
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

}

package hyperparameter.tuning

import Helper.InstanceUtilities.Instance
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD



/**
  * Created by Sophie on 12/2/16.
  * To implement k-fold cross validation so that it can be used to do hyperparameter tuning.
  */

object MyCrossValidation {

  /**
    * Generate k-fold
    * @param data
    * @param k, 1 to 100
    * @return Array of pair of (RDD[Instance], RDD[Instance]), (train, test)
    */
  def kFold(data: RDD[Instance], k: Int = 3): Array[(RDD[Instance], RDD[Instance])] = {
    require(k >= 2 && k <= 100)

    // first persist data into memory
    data.persist()

    val kFolds: Array[RDD[Instance]] = data.randomSplit(Array.fill[Double](k)(1.0))

    val kSplits = (0 until k).toArray.map((i: Int) => ((0 until k).filter(_ != i).toArray.map(kFolds(_)), kFolds(i)))

    val sc: SparkContext = SparkContext.getOrCreate()

    kSplits.map((pair: (Array[RDD[Instance]], RDD[Instance])) => (sc.union(pair._1), pair._2))

//    val numberOfInstances: Long = data.count()
//    val foldSize: Long = numberOfInstances / k
//    val kFoldIndices: Array[Long] = (0 until k - 1).toArray.map((i: Int) => i * foldSize) ++ Array(numberOfInstances)

  }

}

package hyperparameter.tuning

import Helper.InstanceUtilities.Instance
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.storage.StorageLevel



/**
  * Created by Sophie on 12/2/16.
  * To implement k-fold cross validation so that it can be used to do hyperparameter tuning.
  */

object MyCrossValidation {

  /**
    * Generate k-fold
    * @param data
    * @param k, 2 to 20
    * @return Array of pair of (RDD[Instance], RDD[Instance]), (train, test)
    */
  def kFoldRDD(data: RDD[Instance], k: Int = 3): Array[(RDD[Instance], RDD[Instance])] = {
    require(k >= 2 && k <= 20, "k should be between 2 and 20, both inclusive.")

    // first persist data into memory
    if(data.getStorageLevel == StorageLevel.NONE) {
      data.persist(StorageLevel.MEMORY_AND_DISK)
    }

    val kFolds: Array[RDD[Instance]] = data.randomSplit(Array.fill[Double](k)(1.0))

    val kSplits = (0 until k).toArray.map((i: Int) => ((0 until k).filter(_ != i).toArray.map(kFolds(_)), kFolds(i)))

    val sc: SparkContext = SparkContext.getOrCreate()

    kSplits.map((pair: (Array[RDD[Instance]], RDD[Instance])) => (sc.union(pair._1), pair._2))
  }


  def kFoldDF(df: DataFrame, k: Int = 3): Array[(DataFrame, DataFrame)] = {
    require(k >= 2 && k <= 20, "k should be between 2 and 20, both inclusive.")

    // first persist data frame into memory (and disk)
    if(df.rdd.getStorageLevel == StorageLevel.NONE) {
      df.persist(StorageLevel.MEMORY_AND_DISK)
    }

    val kFolds: Array[DataFrame] = df.randomSplit(Array.fill[Double](k)(1.0))

    val kSplits = (0 until k).toArray.map((i: Int) => ((0 until k).filter(_ != i).toArray.map(kFolds(_)), kFolds(i)))

    def _union(dfArr: Array[DataFrame]): DataFrame = {
      require(dfArr.length >= 1, "Union an array of data frames. The length should be at least one.")

      if(dfArr.tail.isEmpty) {
        dfArr.head
      }
      else {
        dfArr.head.union(_union(dfArr.tail))
      }
    }

    kSplits.map((pair: (Array[DataFrame], DataFrame)) => (_union(pair._1), pair._2))
  }

}

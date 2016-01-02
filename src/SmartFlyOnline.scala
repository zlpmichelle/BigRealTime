/**
 * Created by Michelle on 1/6/15.
 */

import org.apache.log4j.{Level, Logger}
import org.apache.spark.streaming.kafka.KafkaUtils
import org.apache.spark.streaming.{Seconds, StreamingContext}
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.linalg.Vectors
import scala.language.reflectiveCalls
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.configuration.{Algo, Strategy}
import org.apache.spark.rdd.RDD



object smartflyOnline {
  def main(args: Array[String]) {
    // set up environment
    val Array(zkQuorum, group, topics, numThreads) = args
    val sparkConf = new SparkConf().setAppName("NaiveBayesSampleCode_Online")
    val ssc =  new StreamingContext(sparkConf, Seconds(2))
    ssc.checkpoint("checkpoint")

    // history flights data
    val data = ssc.sparkContext.textFile("hdfs://172.31.4.45:8020/user/hive/warehouse/p1.db/flight_final_delay/c948d63b0c8f20ad-2fdbcc7af8c7d084_860328325_data.0.")
    val parsedData = data.map(s => s.split(',').map(_.toDouble)).map(x => LabeledPoint(x(0), Vectors.dense(x.tail)))
    val splits = parsedData.randomSplit(Array(0.8, 0.2), seed = 11L)
    val training = splits(0).cache()
    val test = splits(1)

    // scheduled flights data
    val scheduledData = ssc.sparkContext.textFile("hdfs://172.31.4.45:8020/user/hive/warehouse/p1.db/scheduled_final/f64cfeebeddfe667-3bcfd65fba217ec0_930871996_data.0.").map(x => x.split(",").toList).map(x => (x.head, x.tail))


    val categoricalFeaturesInfo = Map(6 -> 19, 7 -> 285, 8 -> 285)
    val maxBins = 286
    val numClasses = 8
    val impurity = "variance"
    val maxDepth = 5

    println("============================== Start Regression Decision Tree Model Training =============================== ")
    val model = DecisionTree.trainRegressor(training, categoricalFeaturesInfo, impurity, maxDepth, maxBins)
    // val model = DecisionTree.trainClassifier(training, numClasses, categoricalFeaturesInfo, impurity, maxDepth, maxBins)
    println("============================== End Modeling =============================== ")
    println("Learned classification Decision Tree model:\n" + model)

    println("============================== Start Decision Tree Training Model Evaluation ============================")
    val labelAndPredsTest = test.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }

    // evaluation of model for test data
    val testMSE = meanSquaredError(model, test)
    println("--------------- Evaluation tests data MSE = " + testMSE + " ---------------")



    // online prediction
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)
    println("-------start----------")
    //StreamingExamples.setStreamingLogLevels()

    val topicMap = topics.split(",").map((_,numThreads.toInt)).toMap
    val lines = KafkaUtils.createStream(ssc, zkQuorum, group, topicMap).map(_._2)

    println("-------incoming line----------")
    lines.print();


    // predict scheduled flights data
    println("============================== Predict Delay Flights based on Decision Tree Training Model ============================")
    val testOnlineScheduledFlight = lines.map{ line => {
      val parts = line.split(',')
      val uniqueFlightID = parts(0)
      val a = parts.tail.map(_.toDouble)
      val b = Vectors.dense(a.toArray)
      val prediction = model.predict(b)
      (uniqueFlightID, prediction)
     }
    }

    testOnlineScheduledFlight.print()

    ssc.start()
    ssc.awaitTermination()

  }

  /**
   * Calculates the mean squared error for regression.
   */
  def meanSquaredError(model: {def predict(features: Vector): Double},
                       data: RDD[LabeledPoint]): Double = {
    data.map { y =>
      val err = model.predict(y.features) - y.label
      err * err
    }.mean()
  }

}


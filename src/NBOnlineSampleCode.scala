import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.streaming.kafka.KafkaUtils
import org.apache.spark.streaming.{Seconds, StreamingContext}
import org.apache.spark.SparkConf

/**
 * Created by Michelle on 15/6/7.
 */
object NBOnlineSampleCode {
  def main(args: Array[String]) {
    // set up environment
    val Array(zkQuorum, group, topics, numThreads) = args
    val sparkConf = new SparkConf().setAppName("NaiveBayesSampleCode_Online")
    val ssc =  new StreamingContext(sparkConf, Seconds(2))
    ssc.checkpoint("checkpoint")

    println("-------start----------")
    val data = ssc.sparkContext.textFile("hdfs://172.31.4.45:8020/user/root/sample_naive_bayes_data.txt")
    val parsedData = data.map { line =>
      val parts = line.split(',')
      LabeledPoint(parts(0).toDouble, Vectors.dense(parts(1).split(' ').map(_.toDouble)))
    }
    // Split data into training (60%) and test (40%).
    val splits = parsedData.randomSplit(Array(0.6, 0.4), seed = 11L)
    val training = splits(0)
    val test = splits(1)

    println("-------train----------")
    val model = NaiveBayes.train(training, lambda = 1.0)
    println("-----Learned regression tree model:\n" + model.toString)

    println("-------prediction----------")
    val predictionAndLabel = test.map(p => (model.predict(p.features), p.label))

    println("-------accuracy----------")
    val accuracy = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / test.count()
    println("-------Accuracy = " + accuracy)



    // online prediction
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)
    println("-------start----------")
    //StreamingExamples.setStreamingLogLevels()

    val topicMap = topics.split(",").map((_,numThreads.toInt)).toMap
    val lines = KafkaUtils.createStream(ssc, zkQuorum, group, topicMap).map(_._2)

    println("-------incoming line----------")
    lines.print();

    val testOnlineData =lines.map{ line =>
     val parts = line.split(',')
      LabeledPoint(parts(0).toDouble, Vectors.dense(parts(1).split(' ').map(_.toDouble)))
    }

    val labelAndPredsOnlineData = testOnlineData.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }

    labelAndPredsOnlineData.print()


    ssc.start()
    ssc.awaitTermination()

  }

}

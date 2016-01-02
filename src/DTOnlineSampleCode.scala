import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.storage.StorageLevel
import org.apache.spark.streaming.kafka.KafkaUtils
import org.apache.spark.streaming.{Minutes, Seconds, StreamingContext}
import org.apache.spark.{SparkContext, SparkConf}

import org.apache.spark.mllib.linalg.Vectors
import scala.language.reflectiveCalls


/**
 * Created by Michelle on 15/6/7.
 */
object DTOnlineSampleCode_Classifier {
  def main(args: Array[String]) {
    // set up environment
    val Array(zkQuorum, group, topics, numThreads) = args
    val sparkConf = new SparkConf().setAppName("DTOnlineSampleCode_Classifier")
    val ssc =  new StreamingContext(sparkConf, Seconds(2))
    ssc.checkpoint("checkpoint")

    if (args.length < 4) {
      System.err.println("Usage: DTOnlineSampleCode <zkQuorum> <group> <topics> <numThreads>")
      System.exit(1)
    }


    // Load and parse the data file.
    val data = MLUtils.loadLibSVMFile(ssc.sparkContext, "hdfs://172.31.4.45:8020/user/root/sample_libsvm_data.txt")
    // Split the data into training and test sets (30% held out for testing)
    val splits = data.randomSplit(Array(0.7, 0.3))
    val (trainingData, testData) = (splits(0), splits(1))

    // Train a DecisionTree model.
    //  Empty categoricalFeaturesInfo indicates all features are continuous.
    val numClasses = 2
    val categoricalFeaturesInfo = Map[Int, Int]()
    val impurity = "gini"
    val maxDepth = 5
    val maxBins = 32

    val model = DecisionTree.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo,
      impurity, maxDepth, maxBins)

    // Evaluate model on test instances and compute test error
    val labelAndPreds = testData.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    val testErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble / testData.count()
    println("Test Error = " + testErr)
    println("Learned classification tree model:\n" + model.toDebugString)


    // online prediction
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)
    println("-------start----------")
    //StreamingExamples.setStreamingLogLevels()

    val topicMap = topics.split(",").map((_,numThreads.toInt)).toMap
    val lines = KafkaUtils.createStream(ssc, zkQuorum, group, topicMap).map(_._2)
    lines.print()

    println("-------online predict----------")

    lines.foreachRDD(rdd =>{
      if(!rdd.isEmpty()) {
        val parsed = rdd.map(_.trim)
          .filter(line => !(line.isEmpty || line.startsWith("#")))
          .map { line =>
          val items = line.split(' ')
          val label = items.head.toDouble
          val (indices, values) = items.tail.filter(_.nonEmpty).map { item =>
            val indexAndValue = item.split(':')
            val index = indexAndValue(0).toInt - 1 // Convert 1-based indices to 0-based.
          val value = indexAndValue(1).toDouble
            (index, value)
          }.unzip
          (label, indices.toArray, values.toArray)
        }

        // Determine number of features.
        val numFeatures = -1
        val d = if (numFeatures > 0) {
          numFeatures
        } else {
          parsed.persist(StorageLevel.MEMORY_ONLY)
          parsed.map { case (label, indices, values) =>
            indices.lastOption.getOrElse(0)
          }.reduce(math.max) + 1
        }

        val testOnlineData = parsed.map { case (label, indices, values) =>
          LabeledPoint(label, Vectors.sparse(d, indices, values))
        }

        val labelAndPredsOnlineData = testOnlineData.map { point =>
          val prediction = model.predict(point.features)
          (point.label, prediction)
        }

        labelAndPredsOnlineData.collect().map(println)
      }
    })

    ssc.start()
    ssc.awaitTermination()
  }

}

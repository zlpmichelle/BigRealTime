import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint

/**
 * Created by Michelle on 15/6/7.
 */

object NaiveBayesSampleCode {
  def main(args: Array[String]) {
    // set up environment
    val conf = new SparkConf()
      .setAppName("NaiveBayesSampleCode")
      .set("spark.executor.memory", "8g")
    val sc = new SparkContext(conf)

    println("-------start----------")
    val data = sc.textFile("hdfs://172.31.4.45:8020/user/root/sample_naive_bayes_data.txt")
    val parsedData = data.map { line =>
      val parts = line.split(',')
      println(parts(0) + "    :   " + parts(1))
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

    val resultPath = "hdfs://172.31.4.45:8020/tmp/naivebayes/result"
    predictionAndLabel.saveAsTextFile(resultPath)
    println("-----saveAsFile: " + resultPath)

    println("-------accuracy----------")
    val accuracy = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / test.count()
    println("-------Accuracy = " + accuracy)
    sc.stop()
  }
}

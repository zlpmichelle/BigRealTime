import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkContext, SparkConf}

/**
 * Created by Michelle on 15/6/7.
 */
object DecisionTreeSampleCode_Classifier {
  def main(args: Array[String]) {
    // set up environment
    val conf = new SparkConf()
      .setAppName("DecisionTreeSampleCode_Classifier")
      .set("spark.executor.memory", "8g")
    val sc = new SparkContext(conf)


    // Load and parse the data file.
    val data = MLUtils.loadLibSVMFile(sc, "hdfs://172.31.4.45:8020/user/root/sample_libsvm_data.txt")
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
    sc.stop()
  }
}

object DecisionTreeSampleCode_Regressor {
  def main(args: Array[String]) {
    // set up environment
    val conf = new SparkConf()
      .setAppName("DecisionTreeSampleCode_Regressor")
      .set("spark.executor.memory", "8g")
    val sc = new SparkContext(conf)


    // Load and parse the data file.
    val data = MLUtils.loadLibSVMFile(sc, "hdfs://172.31.4.45:8020/user/root/sample_libsvm_data.txt")
    // Split the data into training and test sets (30% held out for testing)
    val splits = data.randomSplit(Array(0.7, 0.3))
    val (trainingData, testData) = (splits(0), splits(1))

    // Train a DecisionTree model.
    //  Empty categoricalFeaturesInfo indicates all features are continuous.
    val categoricalFeaturesInfo = Map[Int, Int]()
    val impurity = "variance"
    val maxDepth = 5
    val maxBins = 32

    val model = DecisionTree.trainRegressor(trainingData, categoricalFeaturesInfo, impurity,
      maxDepth, maxBins)

    // Evaluate model on test instances and compute test error
    val labelsAndPredictions = testData.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    val testMSE = labelsAndPredictions.map{ case(v, p) => math.pow((v - p), 2)}.mean()
    println("Test Mean Squared Error = " + testMSE)
    println("Learned regression tree model:\n" + model.toDebugString)
    sc.stop()
  }
}
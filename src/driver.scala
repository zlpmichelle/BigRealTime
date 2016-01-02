import breeze.linalg.DenseMatrix
import breeze.stats.distributions.Gamma
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.mllib.linalg.distributed.RowMatrix

/**
 * Created by Michelle on 15/3/30.
 */
object driver {


  def main(args:Array[String]): Unit ={
    val c = 2000
    val gg = new Gamma(100, 1.0/100)

    val conf = new SparkConf().setMaster("local[4]").setAppName("ss")
    val sc = new SparkContext(conf)

    val vecs = (1 to c).map(gg.sample(c))
              .map(x => Vectors.dense(x))
    val rowRdd = sc.parallelize(vecs)

    val rm = new RowMatrix(rowRdd)
    val re = rm.computePrincipalComponents(10)

    println(re)

  }

}

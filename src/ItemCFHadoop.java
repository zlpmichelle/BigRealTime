/**
 * Created by Michelle on 15/4/9.
 */
import org.apache.hadoop.mapred.JobConf;
import org.apache.mahout.cf.taste.hadoop.item.RecommenderJob;

public class ItemCFHadoop {

    private static final String HDFS = "hdfs://192.168.1.210:9000";

    public static void main(String[] args) throws Exception {
        String inPath = HDFS + "/user/hdfs/userCF";
        String inFile = inPath + "/itemhdfs.csv";
        String outPath = HDFS + "/user/hdfs/userCF/result/";
        String outFile = outPath + "/part-r-00000";
        String tmpPath = HDFS + "/tmp/" + System.currentTimeMillis();

        JobConf conf = config();

        StringBuilder sb = new StringBuilder();
        sb.append("--input ").append(inPath);
        sb.append(" --output ").append(outPath);
        sb.append(" --booleanData true");
        sb.append(" --similarityClassname org.apache.mahout.math.hadoop.similarity.cooccurrence.measures.EuclideanDistanceSimilarity");
        sb.append(" --tempDir ").append(tmpPath);
        args = sb.toString().split(" ");

        RecommenderJob job = new RecommenderJob();
        job.setConf(conf);
        job.run(args);
    }

    public static JobConf config() {
        JobConf conf = new JobConf(ItemCFHadoop.class);
        conf.setJobName("ItemCFHadoop");
        conf.addResource("classpath:/hadoop/core-site.xml");
        conf.addResource("classpath:/hadoop/hdfs-site.xml");
        conf.addResource("classpath:/hadoop/mapred-site.xml");
        return conf;
    }
}

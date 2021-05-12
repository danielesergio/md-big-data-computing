import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import scala.Tuple2;

import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;

/**
 * @author Daniele Sergio 1127732
 */
public class G45HW2 {

    private static Tuple2<Vector, Integer> strToTuple (String str){
        final String[] tokens = str.split(",");
        final double[] data = new double[tokens.length];
        for (int i = 0; i < tokens.length-1; i++) {
            data[i] = Double.parseDouble(tokens[i]);
        }
        final Vector point = Vectors.dense(data);
        final Integer cluster = Integer.valueOf(tokens[tokens.length-1]);
        return new Tuple2<>(point, cluster);
    }

    private static double distancesSum(Vector point, List<Vector> points){
        return points.stream()
                .mapToDouble(currentPoint -> Vectors.sqdist(currentPoint, point))
                .sum();
    }

    public static void main(String[] args) throws IOException {

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // CHECKING NUMBER OF CMD LINE PARAMETERS
        // Parameters are: <path_to_file>, k (integer), t(integer),
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        if (args.length != 3) {
            throw new IllegalArgumentException("USAGE: file_path, k, t");
        }

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // SPARK SETUP
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        final SparkConf conf = new SparkConf(true).setAppName("G45HM1");
        final JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("WARN");

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // INPUT READING
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        // Read input file and subdivide it into K random partitions
        // Each file line line has the following format:  point_x, point_y, cluster_id
        final String filePath = args[0];

        // Read number of clusters k
        //It's not clear how to use this variable. I always use all clusters of the loaded file.
        final int k = Integer.parseInt(args[1]);

        // Read expected sample size per cluster t
        final int t = Integer.parseInt(args[2]);
        //Reads the input set of reviews into an RDD of strings called RawData and subdivides it into K partitions.

        // point 1: Read the input data. In particular, the clustering must be read into an RDD of pairs
        // (point,ClusterID) called fullClustering which must be cached and partitioned into a reasonable number of
        // partitions, e.g., 4-8.
        final JavaPairRDD<Vector,Integer> fullClustering = sc.textFile(filePath)
                .mapToPair(G45HW2::strToTuple)
                .repartition(8)
                .cache();

        // point 2: Compute the size of each cluster and then save the k sizes into an array or list represented by a
        // Broadcast variable named sharedClusterSizes.
        final Broadcast<Map<Integer, Long>> sharedClusterSizes = sc.broadcast(fullClustering
                .values()
                .countByValue());

        // point 3: Extract a sample for each cluster indexed by cluster id
        final Broadcast<Map<Integer,List<Vector>>> clusteringSample = sc.broadcast(fullClustering
                .filter((Function<Tuple2<Vector, Integer>, Boolean>) v1 -> new Random().nextDouble() <= Math.min((double) t /sharedClusterSizes.getValue().get(v1._2),1))
                .mapToPair((PairFunction<Tuple2<Vector, Integer>, Integer, Vector>) Tuple2::swap)
                .groupByKey()
                .mapValues((Function<Iterable<Vector>, List<Vector>>) v1 -> StreamSupport.stream(v1.spliterator(), false).collect(Collectors.toList()))
                .collectAsMap()
        );

        long start = System.currentTimeMillis();
        //point 4: Compute the approximate average silhouette coefficient of the input clustering and assign it to a
        // variable approxSilhFull. (Hint: to do so, you can first transform the RDD fullClustering by mapping each
        // element (point, clusterID) of fullClustering to the approximate silhouette coefficient of 'point').

        //approximate average silhouette coefficient
        double approxSilhFull = fullClustering.map((Function<Tuple2<Vector, Integer>, Double>) clusterPoint -> {
            final int clusterId = clusterPoint._2;

            final double approxAP = distancesSum(clusterPoint._1, clusteringSample.getValue().get(clusterId)) /
                    Math.min(t, sharedClusterSizes.getValue().get(clusterId));

            final double approxBP = clusteringSample.getValue().entrySet().stream()
                    .mapToDouble(entry -> {
                        final int currentClusterId = entry.getKey();
                        if (currentClusterId == clusterId) {
                            return Double.MAX_VALUE;
                        }
                        final List<Vector> samplePoints = entry.getValue();
                        return distancesSum(clusterPoint._1, samplePoints) /
                                Math.min(t, sharedClusterSizes.getValue().get(currentClusterId));
                    })
                    .min()
                    .getAsDouble();

            return (approxBP - approxAP) / Math.max(approxAP, approxBP); //  approximate silhouette coefficient of point
        }).reduce((Function2<Double, Double, Double>) Double::sum) /
                sharedClusterSizes.getValue().values().stream().mapToLong(aLong -> aLong).sum();

        long end = System.currentTimeMillis();
        final long timeApproxSilhFull = end - start;

        //Reordered all clusteringSample point into a list of tuple2(Node, clusterId)
        //The time to reordered the clusteringSample is not included in the computation time of exactSilhSample
        final List<Tuple2<Vector, Integer>> samplePoints = clusteringSample.getValue().entrySet().stream()
                .flatMap((java.util.function.Function<Map.Entry<Integer, List<Vector>>, Stream<Tuple2<Vector, Integer>>>) entry -> entry.getValue().stream().map(
                        (java.util.function.Function<Vector, Tuple2<Vector, Integer>>) vector -> new Tuple2(vector, entry.getKey()))).collect(Collectors.toList());

        start = System.currentTimeMillis();

        //average silhouette coefficient
        //Point 5 Compute (sequentially) the exact silhouette coefficient of the clusteringSample and assign it to a variable exactSilhSample.
        double exactSilhSample = samplePoints.stream().map(clusterPoint -> {
            final int clusterId = clusterPoint._2;

            final List<Vector> clusterPoints = clusteringSample.getValue().get(clusterId);
            final double ap = distancesSum(clusterPoint._1, clusterPoints) / clusterPoints.size();

            final double bp = clusteringSample.getValue().entrySet().stream()
                    .mapToDouble(entry -> {
                        final int otherClusterId = entry.getKey();
                        if(otherClusterId == clusterId){
                            return Double.MAX_VALUE;
                        }
                        final List<Vector> otherClusterPoints = entry.getValue();
                        return distancesSum(clusterPoint._1, otherClusterPoints) / otherClusterPoints.size();
                    })
                    .min()
                    .getAsDouble();

            return (bp - ap) / Math.max(ap, bp); //silhouette coefficient
        }).reduce(0.0, Double::sum) / samplePoints.size();

        end = System.currentTimeMillis();
        final long timeExactSilhSample = end - start;

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // RESULT
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        System.out.printf("Value of approxSilhFull = %.5f%n", approxSilhFull);
        System.out.printf("Time to compute approxSilhFull = %s ms %n", timeApproxSilhFull);
        System.out.printf("Value of exactSilhSample = %.5f%n", exactSilhSample);
        System.out.printf("Time to compute exactSilhSample = %s ms %n", timeExactSilhSample);
    }

}
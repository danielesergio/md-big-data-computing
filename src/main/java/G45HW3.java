import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.rdd.RDD;
import scala.Tuple2;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.function.BiConsumer;
import java.util.function.Consumer;
import java.util.stream.Collectors;
import java.util.stream.StreamSupport;

/**
 * @author Daniele Sergio 1127732
 */
public class G45HW3 {

    private static class Config {
        // A path to a text file containing a point set in Euclidean space. Each line of the file contains the
        // coordinates of a point separated by spaces. (Your program should make no assumptions on the number of dimensions!)
        // Each file line line has the following format:  point_x point_y cluster_id
        final String path;
        // An integer kstart which is the initial number of clusters.
        final int kstart;
        // An integer h which is the number of values of k that the program will test.
        final int h;
        // An integer iter which is the number of iterations of Lloyd's algorithm.
        final int iter;
        // An integer M which is the expected size of the sample used to approximate the silhouette coefficient.
        final int M;
        // An integer L which is the number of partitions of the RDDs containing the input points and their clustering.
        final int L;

        public Config(String path, int kstart, int h, int iter, int m, int l) {
            this.path = path;
            this.kstart = kstart;
            this.h = h;
            this.iter = iter;
            M = m;
            L = l;
        }
    }

    private interface Command<T>{
        T execute();
    }

    private static class TimeCounter{
        private long start;
        void start(){
            start =  System.currentTimeMillis();
        }

        long stop(){
            return  System.currentTimeMillis() - start;
        }

    }

    private static Config readConfig(String[] args){
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // CHECKING NUMBER OF CMD LINE PARAMETERS
        // Parameters are: <path_to_file>, kstart(int), h(int), iter(int), M(int), L(int)
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        if (args.length != 6) {
            throw new IllegalArgumentException("USAGE: file_path, k, t");
        }

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // INPUT READING
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        return new Config(
                args[0],
                Integer.parseInt(args[1]),
                Integer.parseInt(args[2]),
                Integer.parseInt(args[3]),
                Integer.parseInt(args[4]),
                Integer.parseInt(args[5])
        );
    }

    private static JavaSparkContext buildSparkContext(){
        final SparkConf conf = new SparkConf(true).setAppName("G45HM3").setMaster("local[*]");
        final JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("WARN");
        return sc;
    }

    private static Vector fileRowToPoint (String str){
        final String[] tokens = str.split(" ");
        final double[] data = new double[tokens.length];
        for (int i = 0; i < tokens.length; i++) {
            data[i] = Double.parseDouble(tokens[i]);
        }
        final Vector point = Vectors.dense(data);
        return point;
    }

    private static double distancesSum(Vector point, List<Vector> points){
        return points.stream()
                .mapToDouble(currentPoint -> Vectors.sqdist(currentPoint, point))
                .sum();
    }

    private static double approxAvgSilhouette(
            JavaSparkContext sc,
            JavaPairRDD<Vector, Integer> currentClustering,
            double t){
        // Compute the size of each cluster and then save the k sizes into an array or list represented by a
        // Broadcast variable named sharedClusterSizes.
        final Broadcast<Map<Integer, Long>> sharedClusterSizes = sc.broadcast(currentClustering
                .values()
                .countByValue());

        // Extract a sample for each cluster indexed by cluster id
        final Broadcast<Map<Integer,List<Vector>>> clusteringSample = sc.broadcast(currentClustering
                .filter((Function<Tuple2<Vector, Integer>, Boolean>) v1 -> new Random().nextDouble() <= Math.min( t /sharedClusterSizes.getValue().get(v1._2),1))
                .mapToPair((PairFunction<Tuple2<Vector, Integer>, Integer, Vector>) Tuple2::swap)
                .groupByKey()
                .mapValues((Function<Iterable<Vector>, List<Vector>>) v1 -> StreamSupport.stream(v1.spliterator(), false).collect(Collectors.toList()))
                .collectAsMap()
        );
        //Compute the approximate average silhouette coefficient of the input clustering and assign it to a
        // variable approxSilhFull. (Hint: to do so, you can first transform the RDD fullClustering by mapping each
        // element (point, clusterID) of fullClustering to the approximate silhouette coefficient of 'point').

        //approximate average silhouette coefficient
        double a =  currentClustering.map((Function<Tuple2<Vector, Integer>, Double>) clusterPoint -> {
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
        return a;
    }

    public static void main(String[] args) throws IOException {

        final Config config = readConfig(args);

        final JavaSparkContext sc = buildSparkContext();

        final TimeCounter tc = new TimeCounter();
        // point 1: Reads the various parameters passed as command-line arguments. In particular, the set of points
        // must be stored into an RDD called inputPoints, which must be cached and subdivided into L partitions.
        tc.start();

        final JavaRDD<Vector> inputPoints = sc.textFile(config.path)
                .map(G45HW3::fileRowToPoint)
                .repartition(config.L)
                .cache();

        // After reading the parameters print the time spent to read the input points.
        // Spark evaluation is lazy so we must use the data to force the execution
        inputPoints.count();
        System.out.printf("Time for input reading = %s%n", tc.stop());
                
        // point 2.1: Computes a clustering of the input points with k clusters, using the Spark implementation of
        // Lloyd's algorithm described above with iter iterations.
        // The clustering must be stored into an RDD currentClustering of pairs (point, cluster_index) with as many
        // elements as the input points.
        // The RDD must be cached and partitioned into L partitions.
        // (If computed by transforming each element of inputPoints with a map method, it should inherit its partitioning.)

        for (int k = config.kstart; k < (config.kstart + config.h); k++) {

            //fixme first time clustering computation takes more time
            tc.start();
            final JavaPairRDD<Vector, Integer> currentClustering = computeCurrentClustering(config, inputPoints, k);
            final long currentClusteringComputationTime = tc.stop();

            //fixme  first time avg silhouette computation takes more time
            tc.start();
            final double approxAvgSilhouette = approxAvgSilhouette(sc, currentClustering, ((double)config.M) / k);
            final long approxAvgSilhouetteComputationTime = tc.stop();

            System.out.printf("Number of clusters k = %s%n", k);
            System.out.printf("Silhouette coefficient = %s%n", approxAvgSilhouette);
            System.out.printf("Time for clustering = %s%n", currentClusteringComputationTime);
            System.out.printf("Time for silhouette computation = %s%n", approxAvgSilhouetteComputationTime);

        }

    }

    private static JavaPairRDD<Vector, Integer> computeCurrentClustering(Config config, JavaRDD<Vector> inputPoints, int k) {
        final KMeansModel kMeansModel = KMeans.train(inputPoints.rdd(), k, config.iter);

        System.out.printf("Number of centers = %s, %s%n", k, kMeansModel.k());
        final JavaPairRDD<Vector, Integer> currentClustering =
                inputPoints.mapToPair((PairFunction<Vector, Vector, Integer>) vector ->
                        new Tuple2<>(vector, kMeansModel.predict(vector)))
                        .cache();
        //lazy evaluation
        currentClustering.count();
        return currentClustering;
    }


}
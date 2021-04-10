package old;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import scala.Tuple2;

import java.time.Instant;
import java.time.Duration;
import java.util.*;

public class G58HM4
{
    public static void main(String[] args) throws Exception
    {

        //------- PARSING CMD LINE ------------
        // Parameters are:
        // <path to file>, k, L and iter

        if (args.length != 4) {
            System.err.println("USAGE: <filepath> k L iter");
            System.exit(1);
        }
        String inputPath = args[0];
        int k=0, L=0, iter=0;
        try
        {
            k = Integer.parseInt(args[1]);
            L = Integer.parseInt(args[2]);
            iter = Integer.parseInt(args[3]);
        }
        catch(Exception e)
        {
            e.printStackTrace();
        }
        if(k<=2 && L<=1 && iter <= 0)
        {
            System.err.println("Something wrong here...!");
            System.exit(1);
        }
        //------------------------------------
        final int k_fin = k;

        //------- DISABLE LOG MESSAGES
        Logger.getLogger("org").setLevel(Level.OFF);
        Logger.getLogger("akka").setLevel(Level.OFF);

        //------- SETTING THE SPARK CONTEXT
        SparkConf conf = new SparkConf(true).setAppName("kmedian new approach");
        JavaSparkContext sc = new JavaSparkContext(conf);

        //------- PARSING INPUT FILE ------------
        JavaRDD<Vector> pointset = sc.textFile(args[0], L)
                .map(x-> strToVector(x))
                .repartition(L)
                .cache();
        long N = pointset.count();
        System.out.println("Number of points is : " + N);
        System.out.println("Number of clusters is : " + k);
        System.out.println("Number of parts is : " + L);
        System.out.println("Number of iterations is : " + iter);

        //------- SOLVING THE PROBLEM ------------
        double obj = MR_kmedian(pointset, k, L, iter);
        System.out.println("Objective function is : <" + obj + ">");
    }

    private static class DurationStepLogger {
        private Instant start;

        public DurationStepLogger() {
            start = Instant.now();
        }

        private void log(final int step){
            System.out.println(String.format("Stage %s duraration: %s milliseconds", step, Duration.between(start, Instant.now()).toMillis()));
            start = Instant.now();
        }
    }

    public static Double MR_kmedian(JavaRDD<Vector> pointset, int k, int L, int iter)    {
        DurationStepLogger durationStepLogger = new DurationStepLogger();
        //
        // --- ADD INSTRUCTIONS TO TAKE AND PRINT TIMES OF ROUNDS 1, 2 and 3
        //

        //------------- ROUND 1 ---------------------------
        JavaRDD<Tuple2<Vector,Long>> coreset = pointset.mapPartitions(x -> {
            final List<Vector> points = new ArrayList<>();
            final List<Long> weights = new ArrayList<>();
            while (x.hasNext())
            {
                points.add(x.next());
                weights.add(1L);
            }
            final List<Vector> centers = G58HM3.kmeansPP(points, weights, k, iter);
            final List<Long> weight_centers = compute_weights(points, centers);
            final List<Tuple2<Vector,Long>> c_w = new ArrayList<>();
            for(int i =0; i < centers.size(); ++i)
            {
                Tuple2<Vector, Long> entry = new Tuple2<>(centers.get(i), weight_centers.get(i));
                c_w.add(i,entry);
            }
            return c_w.iterator();
        });

        coreset.count();
        durationStepLogger.log(1);
        //------------- ROUND 2 ---------------------------

        final List<Tuple2<Vector, Long>> elems = new ArrayList<>(k*L);
        elems.addAll(coreset.collect());
        final List<Vector> coresetPoints = new ArrayList<>();
        final List<Long> weights = new ArrayList<>();
        for(int i =0; i< elems.size(); ++i){
            coresetPoints.add(i, elems.get(i)._1);
            weights.add(i, elems.get(i)._2);
        }

        final List<Vector> centers = G58HM3.kmeansPP(coresetPoints, weights, k, iter);

        durationStepLogger.log(2);
        //------------- ROUND 3: COMPUTE OBJ FUNCTION --------------------
        //
        //------------- ADD YOUR CODE HERE--------------------------------
        //


        final double result = pointset.glom()
                .mapToPair((PairFunction<List<Vector>, Double, Integer>) vectors -> {
                    //Tuple(avg, numberOfPoints used)
                    return new Tuple2<>(G58HM3.kmeansObj(vectors, centers), vectors.size());
                })
                //aggregate all avg
                .reduce((Function2<Tuple2<Double, Integer>, Tuple2<Double, Integer>, Tuple2<Double, Integer>>) (acc, value) -> {
                    final int numberOfPoints = acc._2 + value._2;
                    return new Tuple2<>( (acc._1 * acc._2 + value._1 * value._2) / numberOfPoints , numberOfPoints);
                })._1;

        durationStepLogger.log(3);

        return result;
    }

    public static final List<Long> compute_weights(final List<Vector> points, final List<Vector> centers){
        Long weights[] = new Long[centers.size()];
        Arrays.fill(weights, 0L);
        for(int i =0; i < points.size(); ++i){
            double tmp = euclidean(points.get(i), centers.get(0));
            int mycenter = 0;
            for(int j = 1; j < centers.size(); ++j){
                if(euclidean(points.get(i),centers.get(j)) < tmp){
                    mycenter = j;
                    tmp = euclidean(points.get(i), centers.get(j));
                }
            }
            weights[mycenter] += 1L;
        }
        final List<Long> fin_weights = new ArrayList<>(Arrays.asList(weights));
        return fin_weights;
    }

    public static Vector strToVector(String str) {
        String[] tokens = str.split(" ");
        double[] data = new double[tokens.length];
        for (int i = 0; i < tokens.length; i++) {
            data[i] = Double.parseDouble(tokens[i]);
        }
        return Vectors.dense(data);
    }

    // Euclidean distance
    public static double euclidean(Vector a, Vector b) {
        return Math.sqrt(Vectors.sqdist(a, b));
    }
}
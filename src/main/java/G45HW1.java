import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.spark.api.java.function.PairFunction;
import scala.Tuple2;

import java.io.IOException;
import java.text.DecimalFormat;
import java.util.stream.StreamSupport;
/**
 * @author Daniele Sergio 1127732
 */
public class G45HW1 {

    private static class Review {
        final String productID;
        final String userID;
        final float rating;
        final Long timestamp;

        private Review(String productID, String userID, float rating, Long timestamp) {
            this.productID = productID;
            this.userID = userID;
            this.rating = rating;
            this.timestamp = timestamp;
        }


        static Review from(String line){
            final String[] lineArray = line.split(",");
            return new Review(lineArray[0],lineArray[1],Float.parseFloat(lineArray[2]), Long.parseLong(lineArray[3]));
        }
    }


    public static void main(String[] args) throws IOException {

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // CHECKING NUMBER OF CMD LINE PARAMETERS
        // Parameters are: K (integer), T(integer), <path_to_file>
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        if (args.length != 3) {
            throw new IllegalArgumentException("USAGE: K T file_path");
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

        // Read number of partitions
        final int K = Integer.parseInt(args[0]);
        // Read input file and subdivide it into K random partitions
        // Each file line line has the following format:  (ProductID,UserID,Rating,Timestamp)
        final String filePath = args[2];

        //Reads the input set of reviews into an RDD of strings called RawData and subdivides it into K partitions.
        final JavaRDD<String> RawData = sc.textFile(filePath).repartition(K).cache();


        // NormalizedRatings contains the pair (ProductID,NormRating), where NormRating=Rating-AvgRating and AvgRating
        // is the average rating of all reviews by the user "UserID".
        // Note that normalizedRatings may contain several pairs for the same product, one for each existing review for
        // that product!
        final JavaPairRDD<String,Float> normalizedRatings = RawData
                .groupBy((Function<String, String>) line -> Review.from(line).userID)
                .flatMapToPair((PairFlatMapFunction<Tuple2<String, Iterable<String>>, String, Float>) stringIterableTuple2 -> {
                    final float avgUserRate = (float) StreamSupport.stream(stringIterableTuple2._2.spliterator(), true)
                            .mapToDouble(line -> Review.from(line).rating)
                            .average()
                            .orElse(0.0);

                    return StreamSupport.stream(stringIterableTuple2._2.spliterator(), true)
                            .map(line -> {
                                final Review review = Review.from(line);
                                return new Tuple2<>(review.productID, review.rating - avgUserRate)  ;
                            }).iterator();
                });


        // maxNormRatings contains exactly one pair (ProductID, MNR) where MNR is the maximum normalized rating of
        // product "ProductID".
        final JavaPairRDD<String,Float> maxNormRatings = normalizedRatings
                .reduceByKey((Function2<Float, Float, Float>) Math::max);

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // RESULT
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        final int T = Integer.parseInt(args[1]);

        System.out.printf("INPUT PARAMETERS: K=%d T=%d file=%s \n\nOUTPUT:%n", K, T, filePath);

        final DecimalFormat decimalFormat = new DecimalFormat("0.0######");

        maxNormRatings
                .mapToPair((PairFunction<Tuple2<String, Float>, Float, String>) Tuple2::swap)
                .sortByKey(false)
                .take(T)
                .forEach(floatStringTuple2 -> System.out.printf("Product %s maxNormRating %s%n", floatStringTuple2._2, decimalFormat.format(floatStringTuple2._1)));

    }

}
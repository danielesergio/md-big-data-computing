import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFunction;
import scala.Tuple2;
import shapeless.Tuple;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.Serializable;
import java.text.DecimalFormat;
import java.time.Instant;
import java.time.Period;
import java.util.*;
import java.util.function.BinaryOperator;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;

import static java.time.temporal.ChronoUnit.MILLIS;
import static java.util.stream.Collectors.groupingBy;
import static java.util.stream.Collectors.toSet;

public class G58HM2 {

    private static final String RESULT_MESSAGE_TEMPLATE = "Max number using %s is: %s";

    public static void main(String[] args) throws FileNotFoundException {
        if (args.length == 0) {
            throw new IllegalArgumentException("Expecting an integer 'k' and the file name on the command line");
        }

        final int k = Integer.parseInt(args[0]);
        final String filePath = args[1];


        // Setup Spark
        SparkConf conf = new SparkConf(true)
                .setAppName("G58HM2");
        JavaSparkContext sc = new JavaSparkContext(conf);

        final JavaRDD<String> docs = sc.textFile(filePath).repartition(k).cache();
        final long docsSize = docs.count();
        final Instant time1 = Instant.now();

        final JavaPairRDD<String, Long> wordsInDocumentUsingAlgorithm1 = wordCount1(docs);

        System.out.println(String.format("Time = %s", MILLIS.between(time1, Instant.now())));

        final JavaRDD<String> distinctWordInDocuments = wordsInDocumentUsingAlgorithm1
                .keys()
                .distinct();

        final double averageLengthOfDistinctWord = distinctWordInDocuments
                .map((Function<String, Integer>) String::length)
                .reduce((Function2<Integer, Integer, Integer>) (v1, v2) -> v1 + v2)
                / (double) distinctWordInDocuments.count();

        System.out.println(String.format("The average length of words is: %s", averageLengthOfDistinctWord));

    }

    private static JavaPairRDD<String, Long> wordCount1Original(JavaRDD<String> docs){
        return docs
                // Map phase
                .flatMapToPair((document) -> {
                    final String[] tokens = document.split(" ");
                    final Map<String, Long> counts = new HashMap<>();
                    final List<Tuple2<String, Long>> pairs = new ArrayList<>();
                    for (String token : tokens) {
                        counts.put(token, 1L + counts.getOrDefault(token, 0L));
                    }
                    for (Map.Entry<String, Long> e : counts.entrySet()) {
                        pairs.add(new Tuple2<>(e.getKey(), e.getValue()));
                    }
                    return pairs.iterator();
                })
                // Reduce phase
                .groupByKey()
                .mapValues((it) -> {
                    long sum = 0;
                    for (long c : it) {
                        sum += c;
                    }
                    return sum;
                });
    }

    private static JavaPairRDD<String, Long> wordCount1(JavaRDD<String> docs){
        return docs
                // Map phase
                .flatMapToPair((document) -> {
                    final String[] tokens = document.split(" ");
                    final Map<String, Long> counts = new HashMap<>();
                    final List<Tuple2<String, Long>> pairs = new ArrayList<>();
                    for (String token : tokens) {
                        counts.put(token, 1L + counts.getOrDefault(token, 0L));
                    }
                    for (Map.Entry<String, Long> e : counts.entrySet()) {
                        pairs.add(new Tuple2<>(e.getKey(), e.getValue()));
                    }
                    return pairs.iterator();
                })
                .reduceByKey((Function2<Long, Long, Long>) (v1, v2) -> v1 + v2);
    }

    //    private static JavaPairRDD<String, Long> wordCount2AZ(JavaRDD<String> docs, int k){
//        return docs
//                //round 1
//                // Map phase
//                .flatMapToPair((document) -> {
//                    final String[] tokens = document.split(" ");
//                    final Map<String,List<String>> map = Arrays.stream(tokens)
//                            .collect(groupingBy(word -> word));
//
//                    final Random random = new Random();
//
//                    return map.entrySet().stream()
//                            .map(entry -> new Tuple2<>(random.nextInt(k), new Tuple2<>(entry.getKey(), entry.getValue().size())))
//                            .iterator();
//
//                })//(x,(w,ci(w)))
//                .groupByKey()
//
//    }
}

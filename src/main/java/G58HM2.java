import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFlatMapFunction;
import scala.Tuple2;

import java.io.IOException;
import java.io.Serializable;
import java.time.Instant;
import java.util.*;
import java.util.stream.StreamSupport;

import static java.time.temporal.ChronoUnit.MILLIS;
import static java.util.stream.Collectors.groupingBy;

public class G58HM2 {

    public static void main(String[] args) throws IOException {
        if (args.length == 0) {
            throw new IllegalArgumentException("Expecting an integer 'k' and the file name on the command line");
        }

        final int k = Integer.parseInt(args[0]);
        final String filePath = args[1];


        // Setup Spark
        SparkConf conf = new SparkConf(true)
                .setAppName("G58HM2");
        JavaSparkContext sc = new JavaSparkContext(conf);
        // FIXME: 06/04/19 remove log configuration before release
        sc.setLogLevel("ERROR");

        final JavaRDD<String> docs = sc.textFile(filePath).repartition(k).cache();
        docs.count();

        WordCounter[] wordCounters = new WordCounter[]{
                new WordCounterWithTimer(new WordCounter1(docs)),
                new WordCounterWithTimer(new WordCounter2a(docs, k)),
                new WordCounterWithTimer(new WordCounter2b(docs))
        };

        System.out.println(String.format("The average length of words is: %s == %s == %s", Arrays.stream(wordCounters).map(WordCounter::count).map(G58HM2::averageLengthOfDistinctWord).toArray()));

//        Scanner scanner = new Scanner(System.in);
//        scanner.nextLine();
    }

    private static double averageLengthOfDistinctWord(JavaPairRDD<String, Long> wordsInDocument) {

        final JavaRDD<String> distinctWordInDocuments = wordsInDocument
                .keys()
                .distinct();

        return distinctWordInDocuments
                .map((Function<String, Integer>) String::length)
                .reduce((Function2<Integer, Integer, Integer>) Integer::sum)
                / (double) distinctWordInDocuments.count();

    }

    private interface WordCounter extends Serializable {
        JavaPairRDD<String, Long> count();

        default Iterator<Tuple2<String, Long>> pairedWordWithNumberOfOccurrences(String document) {
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
        }
    }

    private static class WordCounterWithTimer implements WordCounter {
        private final WordCounter delegate;

        WordCounterWithTimer(WordCounter delegate) {
            this.delegate = delegate;
        }

        @Override
        public JavaPairRDD<String, Long> count() {
            final Instant instant = Instant.now();
            final JavaPairRDD<String,Long> result = delegate.count();
            result.count(); //force execution
            System.out.println(String.format("Time = %s", MILLIS.between(instant, Instant.now())));
            return result;
        }

    }


    private static abstract class AbstractWordCounter implements WordCounter {
        final JavaRDD<String> docs;
        JavaPairRDD<String, Long> wordsInDocument = null;

        AbstractWordCounter(JavaRDD<String> docs) {
            this.docs = docs;
        }

    }

    private static class WordCounter1 extends AbstractWordCounter {
        WordCounter1(JavaRDD<String> docs) {
            super(docs);
        }

        @Override
        public JavaPairRDD<String, Long> count() {
            if(wordsInDocument == null){
                wordsInDocument = docs
                        .flatMapToPair(this::pairedWordWithNumberOfOccurrences)
                        .reduceByKey((Function2<Long, Long, Long>) Long::sum);
            }
            return wordsInDocument;
        }

    }

    private static class WordCounter2a extends AbstractWordCounter{
        private final int k;

        WordCounter2a(JavaRDD<String> docs, int k) {
            super(docs);
            this.k = k;
        }

        @Override
        public JavaPairRDD<String, Long> count() {
            final Random random = new Random();
            return docs
                    //round 1
                    // Map phase
                    .flatMapToPair(this::pairedWordWithNumberOfOccurrences)
                    .groupBy((Function<Tuple2<String, Long>, Integer>) v1 -> random.nextInt(k))
                    .flatMapToPair((PairFlatMapFunction<Tuple2<Integer, Iterable<Tuple2<String, Long>>>, String, Long>) ele -> {
                        final Map<String,List<Tuple2<String,Long>>> map = StreamSupport.stream(ele._2.spliterator(), false)
                                .collect(groupingBy(t -> t._1));

                        return map.values().stream().map(tuple2s -> tuple2s.stream().reduce((stringIntegerTuple2, stringIntegerTuple22) -> new Tuple2<>(stringIntegerTuple2._1, stringIntegerTuple2._2 + stringIntegerTuple22._2)).orElse(null)).iterator();
                    })//round2
                    .reduceByKey((Function2<Long, Long, Long>) Long::sum);

        }
    }

    private static class WordCounter2b extends AbstractWordCounter {

        WordCounter2b(JavaRDD<String> docs) {
            super(docs);
        }

        @Override
        public JavaPairRDD<String, Long> count() {
            return docs
                    //round 1
                    // Map phase
                    .flatMapToPair(this::pairedWordWithNumberOfOccurrences)
                    .mapPartitionsToPair((PairFlatMapFunction<Iterator<Tuple2<String, Long>>, String, Long>) tuple2Iterator -> {
                        Iterable<Tuple2<String, Long>> iterable = () -> tuple2Iterator;
                        final Map<String,List<Tuple2<String,Long>>> map = StreamSupport.stream(iterable.spliterator(), false)
                                .collect(groupingBy(t -> t._1));

                        return map.values().stream().map(tuple2s -> tuple2s.stream().reduce((stringIntegerTuple2, stringIntegerTuple22) -> new Tuple2<>(stringIntegerTuple2._1, stringIntegerTuple2._2 + stringIntegerTuple22._2)).orElse(null)).iterator();

                    },true)
                    .reduceByKey((Function2<Long, Long, Long>) Long::sum);

        }
    }

}

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
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

    //todo dubbi:
    //1 - template (Improved Word count 1) non e' wc1 delle slide -> non ha coppie c(w,1) ma c(w,x) con x >=1
    //2 - cambiando ordine di esecuzione degli algoritmi cambia l'efficenza con cui vengono applicati perche` spark cache automaticamente dei valori parziali;
    //3 - deve ricevere anchee k dalla linea di comando

    public static void main(String[] args) throws IOException {

        // FIXME: 06/04/19 remove log configuration before release
        Logger.getLogger("org").setLevel(Level.OFF);
        Logger.getLogger("akka").setLevel(Level.OFF);


        System.out.println("Insert a integer bigger than 0 for k");
        final Scanner scanner = new Scanner(System.in);
        final int k = scanner.hasNextInt() ? Integer.parseInt(scanner.nextLine()) : 0;
        if(k < 1 ){
            throw new IllegalArgumentException(String.format("k = %s. k must be bigger than 1", k));
        }

        System.out.println("Insert file name with documents to load");
        final String filePath = scanner.hasNextLine() ? scanner.nextLine() : "";

        // Setup Spark
        SparkConf conf = new SparkConf(true)
                .setAppName("G58HM2");
        JavaSparkContext sc = new JavaSparkContext(conf);
        // FIXME: 06/04/19 remove log configuration before release
        sc.setLogLevel("ERROR");

        //load documents into k partition and cached
        final JavaRDD<String> docs = sc.textFile(filePath).repartition(k).cache();
        //because of spark transformations are lazy evaluated it's needed an action to load docs
        docs.count();

        //initialized wordCounters
        WordCounter[] wordCounters = new WordCounter[]{
                new WordCounter1(),
                new WordCounter2a(k),
                new WordCounter2b()
        };

        // FIXME: 07/04/19 spark automatically cache some result so the first algorithm is penalized.
        //        To mitigate this all algorithms are run one before measuring the time
        Arrays.stream(wordCounters).forEach(wc -> wc.count(docs).count());

        //Show the average length of words in documents. The result is calculated three times using as input the result
        //of count method, one for each algorithm. The result should be always the same.
        System.out.println(String.format("The average length of words is: %s == %s == %s", Arrays.stream(wordCounters)
                .map(wc -> new WordCounterWithTimer(wc).count(docs)).map(G58HM2::averageLengthOfDistinctWord).toArray()));
//
//        Scanner scanner = new Scanner(System.in);
//        scanner.nextLine();
    }

    private static double averageLengthOfDistinctWord(JavaPairRDD<String, Long> wordsInDocument) {
        return wordsInDocument
                .keys() //get distinct words in document is guaranteed by design -> wordsInDocument is the result of WordCounter:count() method
                .map((Function<String, Integer>) String::length) // map worth to its length
                .reduce((Function2<Integer, Integer, Integer>) Integer::sum) // sum all lengths
                / (double) wordsInDocument.count(); // divide by the number of words, cast to double to obtain a double division instead of integer division

    }


    /**
     * Implementation of 'Improved Word count 1' using reduceByKey
     */
    private static class WordCounter1 implements WordCounter {

        @Override
        public JavaPairRDD<String, Long> count(JavaRDD<String> docs) {
            return docs
                    .flatMapToPair(this::pairedWordWithNumberOfOccurrences)//map each document to a list of words paired with the number of their occurrences -> (w,ci(w))
                    .reduceByKey((Function2<Long, Long, Long>) Long::sum); //for each set of  pair (w,ci(w)) with the same w generate a pair (w, c(w)) where c(w) is the sum of all ci(w)
        }

    }

    /**
     * Implementation of 'Improved Word count 2' using groupBy
     */
    private static class WordCounter2a extends WordCounter2 {
        private final int k;

        WordCounter2a(int k) {
            this.k = k;
        }

        @Override
        public JavaPairRDD<String, Long> count(JavaRDD<String> docs) {
            final Random random = new Random();
            return docs
                    //round 1
                    // Map phase
                    .flatMapToPair(this::pairedWordWithNumberOfOccurrences)//map each document to a list of words paired with the number of their occurrences -> (w,ci(w))
                    .groupBy((Function<Tuple2<String, Long>, Integer>) v1 -> random.nextInt(k))// group each pair (w,ci(w)) by a random key. key is an integer in [0,k)
                    .flatMapToPair((PairFlatMapFunction<Tuple2<Integer, Iterable<Tuple2<String, Long>>>, String, Long>) ele -> reducePhase1(ele._2))
                    //round2
                    //for each set of pair (w,c(x,w)) with the same w generate a pair (w, c(w)) where c(w) is the sum of all c(x,w)
                    .reduceByKey((Function2<Long, Long, Long>) Long::sum);
        }
    }


    /**
     * Implementation of 'Improved Word count 2' using mapPartitionToPair
     */
    private static class WordCounter2b extends WordCounter2 {

        @Override
        public JavaPairRDD<String, Long> count(JavaRDD<String> docs) {
            return docs
                    //round 1
                    // Map phase
                    .flatMapToPair(this::pairedWordWithNumberOfOccurrences)//map each document to a list of words paired with the number of their occurrences -> (w,ci(w))
                    .mapPartitionsToPair((PairFlatMapFunction<Iterator<Tuple2<String, Long>>, String, Long>) it -> reducePhase1(() -> it),true)
                    //round2
                    //for each word set of (w,ci(w)) with the same w generate a pair (w, c(w)) where c(w) is the sum of all ci(w)
                    .reduceByKey((Function2<Long, Long, Long>) Long::sum);

        }
    }

    /**
     *  Decorator of a WordCounter that calculate and log the time to execute the count method.
     */
    private static class WordCounterWithTimer implements WordCounter {
        private final WordCounter delegate;

        WordCounterWithTimer(WordCounter delegate) {
            this.delegate = delegate;
        }

        @Override
        public JavaPairRDD<String, Long> count(JavaRDD<String> docs) {
            final Instant instant = Instant.now();
            final JavaPairRDD<String,Long> result = delegate.count(docs);
            result.count(); //JavaRDD transformation is lazy so an explicit action it's needed to force a calculation of count method
            System.out.println(String.format("Computation time using %s is: %s", delegate.getClass().getSimpleName(), MILLIS.between(instant, Instant.now())));
            return result;
        }

    }

    /**
     * Abstract class for Wordcounter2 that implement the reduce method of phase 1
     */
    private static abstract class WordCounter2 implements WordCounter{

        /**
         *  For all pairs (x,(w,ci(w))) of group x, and for each word occurring in these pairs produce the
         *  pair(w,c(x,w)) where c(x,w) =∑(x,(w,ci(w))) ci(w).  Now, w is the key for (w,c(x,w))
         */
        Iterator<Tuple2<String,Long>> reducePhase1(Iterable<Tuple2<String, Long>> iterable){
            final Map<String,List<Tuple2<String,Long>>> groupByWord =
                    StreamSupport.stream(iterable.spliterator(), false)
                            .collect(groupingBy(t -> t._1)); //group each pair of collection by word
            //reduce each group of pair to a single pair (w,c(x,w)) -> where w is the word and c(x,w) is the sum of occurrence of w in group x
            return groupByWord.values().stream()
                    .map(wordToOccurrenceList -> wordToOccurrenceList.stream().reduce((wordToOccurrencesAcc, currentWordToOccurrences) -> new Tuple2<>(wordToOccurrencesAcc._1, wordToOccurrencesAcc._2 + currentWordToOccurrences._2)).orElse(null)).iterator();

        }

    }


    /**
     * interface for a Word count algorithm
     */
    private interface WordCounter extends Serializable {
        /**
         * Method that count the occurrences of each words in a collection of documents
         *
         * @param docs as distributed collection of string
         * @return collection of all words in all documents paired with the number of their occurrences.
         *         The result doesn't contain two pairs with the same world.
         */
        JavaPairRDD<String, Long> count(JavaRDD<String> docs);

        //method that take a document and return
        //

        /**
         * Method that paired all the words in a document with the number of occurrences of that word into the document.
         * The result doesn't contain two pairs with the same world.
         */
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
}

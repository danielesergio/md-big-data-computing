import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import scala.Tuple2;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.Serializable;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Scanner;
import java.util.stream.StreamSupport;

/**
 * @author Daniele Sergio
 */
public class G58HM1 {

    private static final String RESULT_MESSAGE_TEMPLATE = "Max number using %s is: %s";

    public static void main(String[] args) throws FileNotFoundException {
        if (args.length == 0) {
            throw new IllegalArgumentException("Expecting the file name on the command line");
        }

        // Read a list of numbers from the program options
        ArrayList<Double> lNumbers = new ArrayList<>();
        Scanner s =  new Scanner(new File(args[0]));
        while (s.hasNext()){
            final double current = Double.parseDouble(s.next());
            if(current<0){
                throw new IllegalArgumentException(String.format("%s is negative. Dataset must contain only nonnegative doubles.", current));
            }
            lNumbers.add(current);
        }
        s.close();

        // Setup Spark
        SparkConf conf = new SparkConf(true)
                .setAppName("Preliminaries");
        JavaSparkContext sc = new JavaSparkContext(conf);

        // Create a parallel collection
        JavaRDD<Double> dNumbers = sc.parallelize(lNumbers);

        // Max number using reduce
        double maxFromReduceMethod = dNumbers.reduce((x, y) -> x > y ? x : y);
        // Max number using max
        double maxFromMaxMethod = dNumbers.max(new DoubleComparatorForSpark());

        System.out.println(String.format(RESULT_MESSAGE_TEMPLATE,"reduce", maxFromReduceMethod));
        System.out.println(String.format(RESULT_MESSAGE_TEMPLATE,"max", maxFromMaxMethod));

        // Normalize numbers
        JavaRDD<Double> dNormalized = dNumbers.map(x -> x / maxFromMaxMethod);

        final DecimalFormat decimalFormatter = new DecimalFormat("#.#");
        final Long minIntervalElements = dNormalized.count() / 10;
        System.out.println("Distributions of number in intervals wide 0.1");
        System.out.println(String.format("Intervals with less of %s elements are ignored",minIntervalElements));

        //Group values by its interval. Each group is wide 0.1: (a,b]. (First interval is [0, 0.1] because its must contain 0).
        dNormalized.groupBy((Function<Double, Double>) v1 -> v1 == 0 ? 0.1 : Math.ceil(v1 * 10) / 10)
                //replace a list of value with its size
                .mapToPair((PairFunction<Tuple2<Double, Iterable<Double>>, Double, Long>) v1 -> new Tuple2<>(v1._1, StreamSupport.stream(v1._2.spliterator(), false).count()))
                //remove interval without enough elements
                .filter((Function<Tuple2<Double, Long>, Boolean>) v1 -> v1._2 >= minIntervalElements)
                //sort pair<Interval,Interval Size> by Interval
                .sortByKey()
                //merge values from all partitions
                .collect()
                .forEach(it -> System.out.println(String.format("Elements in %s%s,%s]: %s ", it._1 == 0.1 ? "[":"(", decimalFormatter.format(it._1 - 0.1), it._1 , it._2)));

    }


    //Class to compare double must implements serializable
    private static class DoubleComparatorForSpark implements Serializable, Comparator<Double> {
        @Override
        public int compare(Double o1, Double o2) {
            return Double.compare(o1,o2);
        }
    }


}

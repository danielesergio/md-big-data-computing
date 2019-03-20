import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Scanner;

/**
 * @author Daniele Sergio
 */
public class G58HM1 {

    private static final String RESULT__MESSAGE_TEMPLATE = "Max number using %s is: %s";

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

        System.out.println(String.format(RESULT__MESSAGE_TEMPLATE,"reduce", maxFromReduceMethod));
        System.out.println(String.format(RESULT__MESSAGE_TEMPLATE,"max", maxFromMaxMethod));

        // Normalize numbers
        JavaRDD<Double> dNormalized = dNumbers.map(x -> x / maxFromMaxMethod);

        // Number bigger than 0,75
        final Long biggerThan75 = dNormalized.filter( x -> x > 0.75 ).count();
        System.out.println(String.format("Numbers bigger than 0,75: %s", biggerThan75));

    }

    //Class to compare double must implements serializable
    private static class DoubleComparatorForSpark implements Serializable, Comparator<Double> {
        @Override
        public int compare(Double o1, Double o2) {
            return Double.compare(o1,o2);
        }
    }


}

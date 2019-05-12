import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

public class G58HM3 {

    public static void main(String[] args) throws IOException {

        final Scanner scanner = new Scanner(System.in);

        System.out.println("Insert number of clusters (k > 0)");
        final int k = scanner.hasNextInt() ? Integer.parseInt(scanner.nextLine()) : 0;
        if(k < 1 ){
            throw new IllegalArgumentException(String.format("k = %s. k must be a positive integer", k));
        }

        System.out.println("Insert number of iterations of Lloyd's (iter >=0)");
        final int iter = scanner.hasNextInt() ? Integer.parseInt(scanner.nextLine()) : 0;
        if(iter < 0 ){
            throw new IllegalArgumentException(String.format("iter = %s. iter must be a non negative integer", iter));
        }

        System.out.println("Insert file name with points to load");
        final String fileName = scanner.hasNextLine() ? scanner.nextLine() : "";

        final List<Vector> P = readVectorsSeq(fileName);
        final List<Long> WP = Collections.nCopies(P.size(), 1L);
        System.out.println("Insert file name with points to load");
        final List<Vector> C = kmeansPP(P, WP,  k, iter);

        System.out.println(String.format("The avarage distance is %s", kmeansObj(P,C)));

    }

    /**
     *
     * @param P a set of points
     * @param WP a set of weigths for P
     * @param k number of clusters
     * @param iter number of iterations of Lloyd's
     * @return  a set C of k centers
     */
    private static List<Vector> kmeansPP(List<Vector> P, List<Long> WP, int k, int iter){
        return null;
    }

    /**
     *
     * @param P  a set of points
     * @param C  set of centerIndexs
     * @return the average distance of a point of P from C
     */
    private static Double kmeansObj(List<Vector> P,List<Vector> C){
        return P.stream().map( p ->
                C.stream().map(c -> calculateDistance(p,c)).min(Double::compareTo).orElseThrow(() -> new IllegalArgumentException("Stream is Empty"))
        ).mapToDouble(Double::doubleValue)
                .average()
                .orElseThrow(() -> new IllegalArgumentException("Stream is Empty"));
    }

    private static double calculateDistance(Vector point, Vector center){
        return Math.sqrt(Vectors.sqdist(point, center));
    }

    private static Vector strToVector(String str) {
        String[] tokens = str.split(" ");
        double[] data = new double[tokens.length];
        for (int i=0; i<tokens.length; i++) {
            data[i] = Double.parseDouble(tokens[i]);
        }
        return Vectors.dense(data);
    }

    private static List<Vector> readVectorsSeq(String filename) throws IOException {
        if (Files.isDirectory(Paths.get(filename))) {
            throw new IllegalArgumentException("readVectorsSeq is meant to read a single file.");
        }
        ArrayList<Vector> result = new ArrayList<>();
        Files.lines(Paths.get(filename))
                .map(G58HM3::strToVector)
                .forEach(result::add);
        return result;
    }
}

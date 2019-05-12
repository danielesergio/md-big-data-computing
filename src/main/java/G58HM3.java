import org.apache.spark.mllib.linalg.BLAS;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class G58HM3 {

    //todo list:
    //1 - capire come calcolare in maniera efficiente il w_p*(d_p)/(sum_{q non center} w_q*(d_q))
    //2 - verificare calcolo distanza sia giusto
    //3 - implmentare Lloyds algorithm
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
        final List<Vector> C = kmeansPP(P, WP,  k, iter);

        System.out.println(String.format("The avarage distance is %s", kmeansObj(P,C)));

    }

    /**
     *
     * @param P a set of points
     * @param WP a set of weigths for points
     * @param k number of clusters
     * @param iter number of iterations of Lloyd's
     * @return  a set C of k centerIndexs
     */
    private static List<Vector> kmeansPP(List<Vector> P, List<Long> WP, int k, int iter){
        final Random random = new Random();
        final int randomIndex = random.nextInt(P.size());
        System.out.println(randomIndex);
        final DistancesFromCenters distancesFromCenters = DistancesFromCenters.newInstance(P, randomIndex, WP);
        for(int i=2; i<k; i++){
            distancesFromCenters.extractNewCenter();
        }
        return lloyd(P, distancesFromCenters.getCenters(), iter);
    }

    /**
     *
     * @param P  a set of points
     * @param C  set of centerIndexs
     * @return the average distance of a point of points from C
     */
    private static Double kmeansObj(List<Vector> P,List<Vector> C){
        return P.stream().map( p ->
                C.stream().map(c -> calculateDistance(p,c)).min(Double::compareTo).orElseThrow(() -> new IllegalArgumentException("Stream is Empty"))
        ).mapToDouble(Double::doubleValue)
                .average()
                .orElseThrow(() -> new IllegalArgumentException("Stream is Empty"));
    }

    private static List<Vector> lloyd(List<Vector> P, List<Vector> initialS, int maxIter){
        boolean stopCondition = false;
        int iter = 0;
        double theta = Double.MAX_VALUE;
        List<Vector> S = initialS;
        while(!stopCondition || iter++ < maxIter){
            List<Partition> partitions = initPartition(P, S);
            for(Partition partition:partitions){
                partition.setCenter(centroid(partition.getPoints()));
            }
            final double newTheta = Partition.kmeans(partitions);
            if( newTheta < theta){
                theta = newTheta;
                S = partitions.stream().map(Partition::getCenter).collect(Collectors.toList());
            } else {
                stopCondition = true;
            }
        }
        return S;
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

    /**
     *
     * @param P pointset
     * @param S S ⊆ points a set of k selected centers
     */
    private static List<Partition> initPartition(List<Vector> P, List<Vector> S){
        final List<Partition> partitions = S.stream().map(s -> new Partition(new ArrayList<>(), s)).collect(Collectors.toList());

        P.stream().filter( p -> !S.contains(p)).forEach(p -> partitions.stream()
                .min(Comparator.comparingDouble(partition -> calculateDistance(partition.getCenter(), p)))
                .orElseThrow(() -> new IllegalArgumentException("Stream is Empty"))
                .addPoint(p));

        return partitions;
    }

    private static Vector centroid(List<Vector> points){
        final Vector centroid = points.stream().reduce(G58HM3::sum).orElseThrow(() -> new IllegalArgumentException(""));
        BLAS.scal(1.0/points.size(), centroid);
        return centroid;
    }

    private static Vector sum(Vector v1, Vector v2){
        return Vectors.dense(IntStream.range(0, v1.size())
                .mapToDouble(i -> v1.apply(i) + v2.apply(i))
                .toArray());
    }

    private static class Partition {

        static double kmeans(List<Partition> partitions){
            return partitions.stream().map(partition -> partition
                    .getPoints()
                    .stream()
                    .map(point -> Vectors.sqdist(point, partition.getCenter()))
                    .reduce(Double::sum).orElseThrow(() -> new IllegalArgumentException("Empty stream")))
                    .reduce(Double::sum).orElseThrow(() -> new IllegalArgumentException("Empty stream"));

        }

        private final List<Vector> points;
        private Vector center;

        public Partition(List<Vector> points, Vector center) {
            this.points = points;
            this.center = center;
        }

        public Vector getCenter() {
            return center;
        }

        public void addPoint(Vector point){
            points.add(point);
        }

        public void setCenter(Vector center) {
            this.center = center;
        }

        public List<Vector> getPoints() {
            return points;
        }
    }

    private static class DistancesFromCenters{
        private final List<Vector> points;
        private Set<Integer> centerIndexs;
        private final Map<Vector, Double> minDistances;
        private List<Integer> listForRandomExtraction;
        List<Long> WP;

        //complexity is O(points.size())
        static DistancesFromCenters newInstance(List<Vector> points, int pointIndex, List<Long> WP){
            final Map<Vector, Double>  minDistances = new HashMap<>(points.size());
            final Vector firstCenter = points.get(pointIndex);
            final Set<Integer> centerIndexes = new LinkedHashSet<>();
            List<Integer> listForRandomExtraction = new ArrayList<>();
            for(int i=0; i<points.size(); i++){
                final Vector point = points.get(i);
                final double distance = calculateDistance(point, firstCenter);
                minDistances.put(point, distance);
                final int weight = (int) Math.floor(WP.get(i) * distance);
                listForRandomExtraction.addAll(Collections.nCopies(weight, i));
            }
            centerIndexes.add(pointIndex);
            return new DistancesFromCenters(points, centerIndexes, minDistances, listForRandomExtraction, WP);
        }

        public DistancesFromCenters(List<Vector> points,
                                    Set<Integer> centerIndexs,
                                    Map<Vector, Double> minDistances,
                                    List<Integer> listForRandomExtraction,
                                    List<Long> WP) {
            this.points = points;
            this.centerIndexs = centerIndexs;
            this.minDistances = minDistances;
            this.listForRandomExtraction = listForRandomExtraction;
            this.WP = WP;
        }

        //complexity is O(points.size())
        void extractNewCenter() {
            final Random random = new Random();
            final int randomIndex =  listForRandomExtraction.get(random.nextInt(listForRandomExtraction.size()));
            final Vector center = points.get(randomIndex);
            centerIndexs.add(randomIndex);
            listForRandomExtraction = new ArrayList<>();
            for(int i = 0; i < points.size(); i++){
                final Vector point = points.get(i);
                final double distance = Math.min(minDistances.get(point), calculateDistance(point, center));
                minDistances.put(point, distance);
                final int weight = (int) Math.floor(WP.get(i) * distance);
                listForRandomExtraction.addAll(Collections.nCopies(weight, i));
            }
        }

        public List<Vector> getCenters() {
            return centerIndexs.stream().map(points::get).collect(Collectors.toList());
        }
    }
}

// 10 - 0 - 635.5578514126418
// 10 - 0 - 635.351697068625
// 10 - 0 - 635.3171721950132
package org.grouplens.lenskit.hello;

import org.apache.commons.math3.linear.DefaultRealMatrixPreservingVisitor;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealMatrixPreservingVisitor;
import org.apache.commons.math3.linear.RealVector;

import java.util.Iterator;

/**
 * Created by jeremyma on 12/08/15.
 */
public class VectorUtils {


    public static int countNonZero(RealVector v){
        int numNonZero = 0;
        Iterator itr = v.iterator();
        while(itr.hasNext()){
            double val = (Double) itr.next();
            if (val == 0.0){
                numNonZero++;
            }
        }
        System.out.println(numNonZero);
        return numNonZero;
    }

    public static int countNonZero(RealMatrix m){

        int numNonZero = 0;
        for (int i=0; i<m.getRowDimension(); i++){
            numNonZero = countNonZero(m.getRowVector(i));
        }

        return numNonZero;
    }

    public static RealMatrix threshold(RealMatrix m, double threshold){
        for (int i=0; i<m.getRowDimension();i++){
            for (int j=0; j<m.getColumnDimension(); j++){
                if (m.getEntry(i,j) < threshold){
                    m.setEntry(i,j,0.0);
                }
            }
        }

        return m;
    }

    /*

    Counts number of values in similarity matrix larger than the threshold value
     */

    public static int countThreshold(RealMatrix m, double threshold){
        int numLarger = 0;
        //System.out.println(threshold);
        for (int i=0; i<m.getRowDimension();i++){
            for (int j=0; j<m.getColumnDimension(); j++){
                if (m.getEntry(i,j) > threshold){
                    numLarger++;
                }
            }
        }
        //System.out.println(numLarger);
        return numLarger;
    }

    public static RealMatrix thresholdSimilarityMatrix(RealMatrix similarity, double threshold_fraction){

        int numIters = 15;
        double lo = 0.0;
        double hi = 1.0;
        double mid = 0.5;

        int totalEntries = similarity.getColumnDimension() * similarity.getRowDimension();
        //System.out.println(totalEntries);
        for (int i=0; i<numIters; i++){
            mid = (lo + hi) / 2.0;

            //copy[copy < mid] = 0.0
            //percent = np.count_nonzero(copy)/float(similarity.shape[0]**2)

            double percent = (double) countThreshold(similarity, mid) / (double) totalEntries;
            /*
            # print percent
            if percent > threshold_fraction:
            # raise the threshold
                    lo = mid
            else:
            hi = mid
            copy = similarity.copy()
            */
            if (percent > threshold_fraction){
                // raise threshold
                lo = mid;
            } else {
                hi = mid;
            }
        }

        similarity = threshold(similarity, mid);

        return similarity;
    }

}

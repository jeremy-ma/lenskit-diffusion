package org.grouplens.lenskit.hello;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.grouplens.lenskit.vectors.similarity.SimilarityDamping;

import javax.inject.Inject;
import java.util.Map;
import java.util.Vector;

/**
 * Created by jeremyma on 11/08/15.
 */
public class ItemUtilityMatrixNormalizer implements UtilityMatrixNormalizer {


    //Create utility matrix, normalize by item mean
    public ItemUtilityMatrixNormalizer(){
    }

    @Override
    public RealMatrix normalizeUtilityMatrix(RealMatrix utility) {
        System.out.println("mean centering utility matrix");
        int numUsers = utility.getRowDimension();
        int numItems = utility.getColumnDimension();
        //calculate global mean
        double totalNumNonZero=0;
        double total=0;

        for (int i=0; i<numUsers;i++){
            RealVector uvector = utility.getRowVector(i);
            totalNumNonZero += (double) VectorUtils.countNonZero(uvector);
            total += uvector.getL1Norm();
        }

        double globalMean = total / totalNumNonZero;

        //mean center the utility matrix (by Item)
        for (int i=0; i<numItems; i++){
            RealVector itemvector = utility.getColumnVector(i);
            double mean = itemvector.getL1Norm() / (double) VectorUtils.countNonZero(itemvector);
            for (int j=0; j<numUsers; j++){
                double entry = itemvector.getEntry(j);
                if (entry > 0.0){
                    itemvector.setEntry(j,entry-mean-globalMean);
                }
            }
            utility.setColumnVector(i, itemvector);
        }

        return utility;
    }
}





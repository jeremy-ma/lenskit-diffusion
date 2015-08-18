package org.grouplens.lenskit.hello;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

/**
 * Created by jeremyma on 13/08/15.
 */
public class UserUtilityMatrixNormalizer implements UtilityMatrixNormalizer {

    /*
    normalize utility matrix by user mean
     */
    @Override
    public RealMatrix normalize(RealMatrix utility) {
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

        //mean center the utility matrix (by user)
        for (int i=0; i<numUsers; i++){
            RealVector uvector = utility.getRowVector(i);
            double mean = uvector.getL1Norm() / (double) VectorUtils.countNonZero(uvector);
            for (int j=0; j<numItems; j++){
                double entry = uvector.getEntry(j);
                if (entry > 0.0){
                    uvector.setEntry(j, entry - mean - globalMean);
                }
            }
            utility.setRowVector(i, uvector);
        }

        return utility;
    }
}

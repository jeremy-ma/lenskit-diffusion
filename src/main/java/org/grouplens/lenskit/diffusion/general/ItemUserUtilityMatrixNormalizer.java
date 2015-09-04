package org.grouplens.lenskit.diffusion.general;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import java.util.ArrayList;

/**
 * removes the global, user and item means from the utility matrix
 * TODO: investigate effect of using least squares normalization
 * Created by jeremyma on 4/09/15.
 */
public class ItemUserUtilityMatrixNormalizer implements UtilityMatrixNormalizer {
    @Override
    public RealMatrix normalize(RealMatrix utility) {
        System.out.println("mean centering utility matrix");
        int numUsers = utility.getRowDimension();
        int numItems = utility.getColumnDimension();
        ArrayList<Double> itemMeans = new ArrayList<Double>(numItems);
        ArrayList<Double> userMeans = new ArrayList<Double>(numUsers);

        //calculate global mean
        double totalNumNonZero=0;
        double total=0;

        for (int i=0; i<numUsers;i++){
            RealVector uvector = utility.getRowVector(i);
            totalNumNonZero += (double) VectorUtils.countNonZero(uvector);
            total += uvector.getL1Norm();
        }

        double globalMean = total / totalNumNonZero;

        //collect the item means
        for (int i=0; i<numItems; i++){
            RealVector itemvector = utility.getColumnVector(i);
            double mean = itemvector.getL1Norm() / (double) VectorUtils.countNonZero(itemvector);
            itemMeans.set(i,mean);

        }

        //collect the user means
        for (int i=0; i<numUsers; i++){
            RealVector uservector = utility.getRowVector(i);
            double mean = uservector.getL1Norm() / (double) VectorUtils.countNonZero(uservector);
            userMeans.set(i,mean);
        }

        //TODO: do this with the efficient matrix visitor thing, OK for now
        for (int i=0; i<numItems; i++){
            for (int j=0; j<numUsers; j++){
                utility.setEntry(i,j,utility.getEntry(i,j) - globalMean - itemMeans.get(i) - userMeans.get(j));
            }
        }

        return utility;
    }
}

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
            itemMeans.add(i,mean-globalMean);

        }

        //collect the user means
        for (int i=0; i<numUsers; i++){
            RealVector uservector = utility.getRowVector(i);
            double mean = uservector.getL1Norm() / (double) VectorUtils.countNonZero(uservector);
            userMeans.add(i,mean-globalMean);
        }

        //TODO: do this with the efficient matrix visitor thing, OK for now
        //subtract the baseline predictor
        for (int i=0; i<numUsers; i++){
            for (int j=0; j<numItems; j++){
                utility.setEntry(i,j,utility.getEntry(i,j) - globalMean - itemMeans.get(j) - userMeans.get(i));
            }
        }

        return utility;
    }
}

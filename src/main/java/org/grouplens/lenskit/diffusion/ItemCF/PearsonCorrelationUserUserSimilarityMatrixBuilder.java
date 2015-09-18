package org.grouplens.lenskit.diffusion.ItemCF;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.stat.correlation.PearsonsCorrelation;

/**
 * Created by jeremyma on 18/09/15.
 */
public class PearsonCorrelationUserUserSimilarityMatrixBuilder implements UserUserSimilarityMatrixBuilder {

    @Override
    public RealMatrix build(RealMatrix utility) {
        PearsonsCorrelation pearson = new PearsonsCorrelation();

        int numUsers = utility.getRowDimension();

        RealMatrix similarity = MatrixUtils.createRealMatrix(numUsers, numUsers);

        // create adjusted cosine similarity matrix
        for (int i=0; i<numUsers; i++){
            for (int j=i; j<numUsers; j++){
                double simVal = 0.0;
                if (utility.getRowVector(i).getNorm() != 0 && utility.getRowVector(j).getNorm() != 0) {
                    simVal = pearson.correlation(utility.getRowVector(i).toArray(),utility.getRowVector(j).toArray());
                }
                similarity.setEntry(i,j,simVal);
                similarity.setEntry(j,i,simVal);
            }
        }
        return similarity;
    }

}

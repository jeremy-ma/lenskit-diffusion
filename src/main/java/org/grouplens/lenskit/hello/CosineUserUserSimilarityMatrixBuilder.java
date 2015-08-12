package org.grouplens.lenskit.hello;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

/**
 * Created by jeremyma on 12/08/15.
 */
public class CosineUserUserSimilarityMatrixBuilder implements UserUserSimilarityMatrixBuilder {

    @Override
    public RealMatrix build(RealMatrix utility) {
        int numUsers = utility.getRowDimension();

        RealMatrix similarity = MatrixUtils.createRealMatrix(numUsers, numUsers);

        // create adjusted cosine similarity matrix
        for (int i=0; i<numUsers; i++){
            for (int j=i; j<numUsers; j++){
                double simVal = 0.0;
                if (utility.getRowVector(i).getNorm() != 0 && utility.getRowVector(j).getNorm() != 0) {
                    simVal = utility.getRowVector(i).cosine(utility.getRowVector(j));
                }
                similarity.setEntry(i,j,simVal);
                similarity.setEntry(j,i,simVal);
            }
        }

        return similarity;
    }
}

package org.grouplens.lenskit.diffusion.UserCF;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.stat.correlation.PearsonsCorrelation;

/**
 * Created by jeremyma on 23/09/15.
 */
public class PearsonCorrelationItemItemSimilarityMatrixBuilder implements ItemItemSimilarityMatrixBuilder {
    @Override
    public RealMatrix build(RealMatrix utility) {
        int numItems = utility.getColumnDimension();
        PearsonsCorrelation pearson = new PearsonsCorrelation();


        RealMatrix similarity = MatrixUtils.createRealMatrix(numItems, numItems);

        // create adjusted cosine similarity matrix
        for (int i=0; i<numItems; i++){
            for (int j=i; j<numItems; j++){
                double simVal = 0.0;
                if (utility.getColumnVector(i).getNorm() != 0 && utility.getColumnVector(j).getNorm() != 0) {
                    simVal = pearson.correlation(utility.getColumnVector(i).toArray(),utility.getColumnVector(j).toArray());
                }
                similarity.setEntry(i,j,simVal);
                similarity.setEntry(j,i,simVal);
            }
        }

        return similarity;
    }
}

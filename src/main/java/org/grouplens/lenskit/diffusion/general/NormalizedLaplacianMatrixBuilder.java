package org.grouplens.lenskit.diffusion.general;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

/**
 * Created by jeremyma on 12/08/15.
 */
public class NormalizedLaplacianMatrixBuilder implements LaplacianMatrixBuilder {

    @Override
    public RealMatrix build(RealMatrix similarity, double alpha) {
        RealMatrix laplacian = similarity.copy();
        for (int i=0; i<similarity.getRowDimension(); i++){
            RealVector v = laplacian.getRowVector(i);
            double degree = v.getL1Norm(); //actually Dii + 1.0 but cancels in next step
            v.mapMultiplyToSelf(-1.0);
            v.addToEntry(i, degree);
            //normalize
            if (degree > 0.0){
                v.mapDivideToSelf(degree);
            } else {
                v.setEntry(i,1.0);
            }
            laplacian.setRowVector(i,v);
        }
        return laplacian.scalarMultiply(alpha);
    }
}

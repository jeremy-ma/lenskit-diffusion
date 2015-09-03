package org.grouplens.lenskit.hello;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

/**
 * Created by jeremyma on 12/08/15.
 *
 * Computes the outdegree laplacian matrix and scales by alpha
 */
public class RegularLaplacianMatrixBuilder implements LaplacianMatrixBuilder {

    @Override
    public RealMatrix build(RealMatrix similarity, double alpha) {
        RealMatrix laplacian = similarity.copy();
        int dimension = similarity.getRowDimension();
        for (int i=0; i<dimension; i++){
            RealVector v = laplacian.getRowVector(i); //ensures outdegree laplacian
            double degree = v.getL1Norm(); //actually Dii + degree but cancels in next step
            v.mapMultiplyToSelf(-1.0);
            v.addToEntry(i, degree);

            laplacian.setRowVector(i,v);
        }

        double diagonalSum = 0;


        //convert alpha from normalized to unnormalized
        for (int i=0; i<dimension; i++){
            diagonalSum += laplacian.getEntry(i,i);
        }

        double ratio_diagL_diagNL = diagonalSum / (double) dimension;
        double alpha_L = alpha / ratio_diagL_diagNL;
        return laplacian.scalarMultiply(alpha_L);
    }
}
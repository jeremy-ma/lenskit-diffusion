package org.grouplens.lenskit.hello;

import com.jmatio.io.MatFileReader;
import com.jmatio.types.MLDouble;
import org.apache.commons.lang3.builder.Diff;
import org.apache.commons.math3.linear.MatrixUtils;
import org.grouplens.lenskit.vectors.SparseVector;
import org.grouplens.lenskit.vectors.VectorEntry;
import org.grouplens.lenskit.vectors.similarity.VectorSimilarity;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

import javax.inject.Inject;

/**
 * Created by jeremyma on 22/04/15.
 */
public class DiffusionDistanceVectorSimilarity implements VectorSimilarity {
    private RealMatrix diffMatrix = null;

    @Inject
    public DiffusionDistanceVectorSimilarity(){
        //read in the matrix
        try{
            MatFileReader reader = new MatFileReader("ml100k_diffusion_norm.mat");
            MLDouble red = (MLDouble) reader.getMLArray("ml100k_diffusion_norm");
            double [][] diffusion = red.getArray();
            this.diffMatrix = MatrixUtils.createRealMatrix(diffusion);
            System.out.println("Matrix is made");

        } catch (Exception e){
            System.out.println(e.getMessage());
            System.out.println("Failed to read in the diffusion matrix");
        }
    }

    /**
     * Computes the similarity as
     * (1-|v1-v2|_2) where v1 and v2 are the diffused vectors normalized to be unit vectors
     * @param vec1
     * @param vec2
     * @return Similarity in the range [-1,1]
     */
    @Override
    public double similarity(SparseVector vec1, SparseVector vec2) {
        ArrayRealVector v1 = new ArrayRealVector(1682);
        ArrayRealVector v2 = new ArrayRealVector(1682);

        for (VectorEntry v:vec1){
            v1.setEntry((int)v.getKey()-1,v.getValue());
        }
        for (VectorEntry v:vec2){
            try{
                v2.setEntry((int)v.getKey()-1,v.getValue());
            } catch (Exception e){
                System.out.println(v.getKey());
                throw new IllegalArgumentException("asdf");
            }

        }
        ArrayRealVector v_diff = (ArrayRealVector) this.diffMatrix.preMultiply(v1);
        ArrayRealVector w_diff = (ArrayRealVector) this.diffMatrix.preMultiply(v2);

        v_diff = (ArrayRealVector) v_diff.unitVector();
        w_diff = (ArrayRealVector) w_diff.unitVector();

        //System.out.println((1.37-v_diff.subtract(w_diff).getNorm())*5);

        return ((1.37-v_diff.subtract(w_diff).getNorm())*5);

    }

    @Override
    public boolean isSparse() {
        return false;
    }

    @Override
    public boolean isSymmetric() {
        return true;
    }
}

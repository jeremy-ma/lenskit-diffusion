package org.grouplens.lenskit.hello;

import com.jmatio.io.MatFileReader;
import com.jmatio.types.MLDouble;
import org.apache.commons.math3.exception.OutOfRangeException;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.grouplens.lenskit.vectors.SparseVector;
import org.grouplens.lenskit.vectors.similarity.VectorSimilarity;
import org.grouplens.lenskit.vectors.VectorEntry;

import javax.inject.Inject;

/**
 * Created by jeremyma on 19/04/15.
 */
public class DiffusionCosineVectorSimilarity implements VectorSimilarity {
    private RealMatrix diffMatrix = null;

    @Inject
    public DiffusionCosineVectorSimilarity(){
        //read in the matrix
        try{
            MatFileReader reader = new MatFileReader("/Users/jeremyma/Documents/Research/ml100k_diffusion_norm.mat");
            MLDouble red = (MLDouble) reader.getMLArray("ml100k_diffusion_norm");
            double [][] diffusion = red.getArray();
            this.diffMatrix = MatrixUtils.createRealMatrix(diffusion);
            System.out.println("Matrix is made");

        } catch (Exception e){
            System.out.println(e.getMessage());
            System.out.println("Failed to read in the diffusion matrix");
        }

    }

    public boolean isSparse(){
        return false;
    }

    public double similarity(SparseVector vec1, SparseVector vec2){

        //System.out.println(vec1);
        //System.out.println(vec2);
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

        double sim = v_diff.cosine(w_diff);

        System.out.println(sim);

        return sim;

    }

    public boolean isSymmetric(){
        return true;
    }

}

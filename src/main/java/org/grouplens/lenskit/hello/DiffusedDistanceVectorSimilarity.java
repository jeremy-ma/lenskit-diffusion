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

import java.lang.reflect.Array;
import java.util.HashMap;

import javax.inject.Inject;

/**
 * Created by jeremyma on 22/04/15.
 */
public class DiffusedDistanceVectorSimilarity implements VectorSimilarity {
    private RealMatrix diffMatrix = null;
    private HashMap<SparseVector, ArrayRealVector> cache = null;
    @Inject
    public DiffusedDistanceVectorSimilarity(@DiffusionMatrixType String diffusionFileName){
        //read in the matrix
        try{
            MatFileReader reader = new MatFileReader(diffusionFileName);
            MLDouble red = (MLDouble) reader.getMLArray("diffusion");
            double [][] diffusion = red.getArray();
            this.diffMatrix = MatrixUtils.createRealMatrix(diffusion);
            System.out.println("Matrix is made");
        } catch (Exception e){
            System.out.println(e.getMessage());
            System.out.println("Failed to read in the diffusion matrix");
        }
        //initialise the cache
        cache = new HashMap<SparseVector, ArrayRealVector>();
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

        ArrayRealVector v_diff;
        ArrayRealVector w_diff;

        if (( v_diff = this.cache.get(vec1) ) == null){
            v_diff = this.getDiffused(vec1);
            this.cache.put(vec1,v_diff);
        }

        if (( w_diff = this.cache.get(vec2) ) == null){
            w_diff = this.getDiffused(vec2);
            this.cache.put(vec2,w_diff);
        }

        if (v_diff.getNorm() > 0) {
            v_diff = (ArrayRealVector) v_diff.unitVector();
        }
        if (w_diff.getNorm() >0){
            w_diff = (ArrayRealVector) w_diff.unitVector();
        }

        //System.out.println((1.37-v_diff.subtract(w_diff).getNorm())*5);
        ArrayRealVector v_w = v_diff.subtract(w_diff);
        double distance = v_diff.subtract(w_diff).getNorm();

        if (distance != v_w.getNorm()){
            System.out.println("Ooops!");
        }

        //return ((1.37-5*distance));
        return 1-distance;
    }

    private ArrayRealVector getDiffused(SparseVector v){
        ArrayRealVector w = new ArrayRealVector(this.diffMatrix.getColumnDimension());

        for (VectorEntry entry:v){
            w.setEntry((int) entry.getKey() - 1, entry.getValue());
        }

        ArrayRealVector w_diff = (ArrayRealVector) this.diffMatrix.preMultiply(w);

        return w_diff;

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

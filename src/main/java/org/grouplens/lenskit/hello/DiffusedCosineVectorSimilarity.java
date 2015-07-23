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
import java.util.HashMap;

/**
 * Created by jeremyma on 19/04/15.
 */
public class DiffusedCosineVectorSimilarity implements VectorSimilarity {
    private RealMatrix diffMatrix = null;
    private HashMap<SparseVector, ArrayRealVector> cache = null;

    @Inject
    public DiffusedCosineVectorSimilarity(DiffusionModel model){
        //read in the matrix
        diffMatrix = model.getDiffusionMatrix();
        cache = new HashMap<SparseVector, ArrayRealVector>();
    }

    public boolean isSparse(){
        return false;
    }

    public double similarity(SparseVector vec1, SparseVector vec2){

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
        if (w_diff.getNorm() > 0) {
            w_diff = (ArrayRealVector) w_diff.unitVector();
        }

        return v_diff.cosine(w_diff);

    }

    private ArrayRealVector getDiffused(SparseVector v){
        ArrayRealVector w = new ArrayRealVector(this.diffMatrix.getColumnDimension());
        for (VectorEntry entry:v){
            w.setEntry((int) entry.getKey() - 1, entry.getValue());
        }

        ArrayRealVector w_diff = (ArrayRealVector) this.diffMatrix.preMultiply(w);

        return w_diff;

    }

    public boolean isSymmetric(){
        return true;
    }

}

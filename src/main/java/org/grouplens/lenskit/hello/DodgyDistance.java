package org.grouplens.lenskit.hello;

import it.unimi.dsi.fastutil.longs.LongSet;
import org.grouplens.lenskit.collections.LongUtils;
import org.grouplens.lenskit.core.Shareable;
import org.grouplens.lenskit.vectors.MutableSparseVector;
import org.grouplens.lenskit.vectors.SparseVector;
import org.grouplens.lenskit.vectors.similarity.VectorSimilarity;

import javax.inject.Inject;
import java.io.Serializable;

@Shareable
public class DodgyDistance implements VectorSimilarity, Serializable {
    private static final long serialVersionUID = 1L;


    /**
     * Construct a new distance similarity function.
     * It computes similarity as (1-|v1-v2|_2). after normalizing vectors to be unit vectors
     * Similarity is in range [-1,1];
     */
    @Inject
    public DodgyDistance() {
    }

    @Override
    public double similarity(SparseVector vec1, SparseVector vec2) {
        final double distance;
        // One of the vector is empty
        if (vec1.norm() == 0 || vec2.norm() == 0){
            return Double.NaN;
        }

        LongSet ts = LongUtils.setUnion(vec1.keySet(), vec2.keySet());

        MutableSparseVector v1 = MutableSparseVector.create(ts);
        v1.fill(0);
        v1.set(vec1);
        v1.multiply(1.0 / v1.norm());
        v1.addScaled(vec2, -1.0 / vec2.norm());

        distance = v1.norm();
        return 1.37-distance*5;
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
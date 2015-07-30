package org.grouplens.lenskit.hello;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.grouplens.lenskit.core.Shareable;
import org.grouplens.lenskit.knn.item.ItemSimilarity;
import org.grouplens.lenskit.knn.user.UserSimilarity;
import org.grouplens.lenskit.vectors.SparseVector;
import org.grouplens.lenskit.vectors.similarity.VectorSimilarity;

import javax.inject.Inject;
import java.io.Serializable;

/**
 * Created by jeremyma on 27/07/15.
 */
@Shareable
public class PrediffusedItemCosineSimilarity implements ItemSimilarity, Serializable{

    private static final long serialVersionUID = 1L;
    private RealMatrix diffusedUtility = null;

    @Inject
    public PrediffusedItemCosineSimilarity(DiffusedUtilityModel model) {
        diffusedUtility = model.getDiffusionMatrix();
    }

    @Override
    public double similarity(long i1, SparseVector v1, long i2, SparseVector v2) {
        RealVector u1 = null;
        RealVector u2 = null;

        u1 = diffusedUtility.getColumnVector((int) i1 - 1);
        u2 = diffusedUtility.getColumnVector((int) i2 - 1);
        return u1.cosine(u2);
    }

    @Override
    public boolean isSparse() {
        return false;
    }

    @Override
    public boolean isSymmetric() {
        return true;
    }

    @Override
    public String toString() {
        return "{similarity: " + "precomputed diffused utility cosine similarity" + "}";
    }
}

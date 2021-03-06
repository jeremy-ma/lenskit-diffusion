package org.grouplens.lenskit.diffusion.org.grouplens.lenskit.diffusion.unused;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.grouplens.lenskit.core.Shareable;
import org.grouplens.lenskit.diffusion.precomputed.DiffusedUtilityModel;
import org.grouplens.lenskit.knn.user.UserSimilarity;
import org.grouplens.lenskit.vectors.SparseVector;

import javax.inject.Inject;
import java.io.Serializable;

/**
 * Created by jeremyma on 27/07/15.
 */
@Shareable
public class PrediffusedUserCosineSimilarity implements UserSimilarity, Serializable{

    private static final long serialVersionUID = 1L;
    private RealMatrix diffusedUtility = null;

    @Inject
    public PrediffusedUserCosineSimilarity(DiffusedUtilityModel model) {
        diffusedUtility = model.getDiffusionMatrix();
    }

    @Override
    public double similarity(long i1, SparseVector v1, long i2, SparseVector v2) {
        RealVector u1 = null;
        RealVector u2 = null;

        u1 = diffusedUtility.getRowVector((int) i1 - 1);
        u2 = diffusedUtility.getRowVector((int) i2 - 1);
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

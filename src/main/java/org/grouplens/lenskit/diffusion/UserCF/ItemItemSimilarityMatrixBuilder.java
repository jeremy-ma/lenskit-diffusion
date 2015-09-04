package org.grouplens.lenskit.diffusion.UserCF;

import org.apache.commons.math3.linear.RealMatrix;

/**
 * Created by jeremyma on 3/09/15.
 */
public interface ItemItemSimilarityMatrixBuilder {
    public RealMatrix build(RealMatrix utility);
}

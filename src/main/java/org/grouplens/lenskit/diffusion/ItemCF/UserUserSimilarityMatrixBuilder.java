package org.grouplens.lenskit.diffusion.ItemCF;

import org.apache.commons.math3.linear.RealMatrix;
import org.grouplens.grapht.annotation.DefaultImplementation;
import org.grouplens.lenskit.diffusion.ItemCF.CosineUserUserSimilarityMatrixBuilder;

/**
 * Created by jeremyma on 12/08/15.
 */

@DefaultImplementation(CosineUserUserSimilarityMatrixBuilder.class)
public interface UserUserSimilarityMatrixBuilder {
    public RealMatrix build(RealMatrix utility);
}

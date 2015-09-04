package org.grouplens.lenskit.diffusion.general;

import org.apache.commons.math3.linear.RealMatrix;
import org.grouplens.grapht.annotation.DefaultImplementation;
import org.grouplens.lenskit.diffusion.general.ItemUtilityMatrixNormalizer;

/**
 * Created by jeremyma on 11/08/15.
 */

@DefaultImplementation(ItemUtilityMatrixNormalizer.class)
public interface UtilityMatrixNormalizer {

    //normalize the utility matrix
    public RealMatrix normalize(RealMatrix utility);
}

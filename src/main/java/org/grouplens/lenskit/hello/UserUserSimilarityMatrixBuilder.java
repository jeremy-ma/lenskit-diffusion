package org.grouplens.lenskit.hello;

import org.apache.commons.math3.linear.RealMatrix;
import org.grouplens.grapht.annotation.DefaultImplementation;

/**
 * Created by jeremyma on 12/08/15.
 */

@DefaultImplementation(CosineUserUserSimilarityMatrixBuilder.class)

public interface UserUserSimilarityMatrixBuilder {
    public RealMatrix build(RealMatrix utility);
}

package org.grouplens.lenskit.diffusion.general;

import org.apache.commons.math3.linear.RealMatrix;
import org.grouplens.grapht.annotation.DefaultImplementation;

/**
 * Created by jeremyma on 12/08/15.
 */

@DefaultImplementation(NormalizedLaplacianMatrixBuilder.class)
public interface LaplacianMatrixBuilder {

    public RealMatrix build(RealMatrix similarity, double alpha);

}

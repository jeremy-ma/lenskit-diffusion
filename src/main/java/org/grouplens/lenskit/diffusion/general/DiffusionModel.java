package org.grouplens.lenskit.diffusion.general;

import org.apache.commons.math3.linear.RealMatrix;
import org.grouplens.grapht.annotation.DefaultProvider;
import org.grouplens.lenskit.core.Shareable;
import org.grouplens.lenskit.diffusion.ItemCF.ItemCFDiffusionModel;


public interface DiffusionModel {
    /**
     * Get the diffusion matrix.
     * @return The Diffusion Matrix.
     */
    public RealMatrix getDiffusionMatrix();

}

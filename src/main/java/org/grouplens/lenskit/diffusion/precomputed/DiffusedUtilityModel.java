package org.grouplens.lenskit.diffusion.precomputed;

import org.apache.commons.math3.linear.RealMatrix;
import org.grouplens.grapht.annotation.DefaultProvider;

/**
 *
 * Made by jeremy
 *
 */
@DefaultProvider(DiffusedUtilityModelBuilder.class)
public class DiffusedUtilityModel {
    private final RealMatrix diffusedUtilityMatrix;

    public DiffusedUtilityModel(RealMatrix diffusedUtilityMatrix) {
        this.diffusedUtilityMatrix = diffusedUtilityMatrix;
    }

    /**
     * Get the diffusion matrix.
     * @return The Diffusion Matrix.
     */
    public RealMatrix getDiffusionMatrix() {
        return this.diffusedUtilityMatrix;
    }

}

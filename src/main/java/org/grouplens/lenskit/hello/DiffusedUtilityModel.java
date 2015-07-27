package org.grouplens.lenskit.hello;

import org.apache.commons.math3.linear.RealMatrix;
import org.grouplens.grapht.annotation.DefaultProvider;
import org.grouplens.lenskit.core.Shareable;

import java.io.Serializable;

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

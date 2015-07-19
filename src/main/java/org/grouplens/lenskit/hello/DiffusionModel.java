package org.grouplens.lenskit.hello;

import org.apache.commons.math3.linear.RealMatrix;
import org.grouplens.grapht.annotation.DefaultProvider;
import org.grouplens.lenskit.core.Shareable;
import org.grouplens.lenskit.vectors.ImmutableSparseVector;
import org.grouplens.lenskit.vectors.SparseVector;

import java.io.Serializable;

/**
 * A 'model' object storing the precomputed item biases.  The {@link DefaultProvider} annotation
 * specifies how this model will be built: it will be built using the model builder class.  The
 * {@link Shareable} annotation tells LensKit that the model can be reused between different
 * recommenders.
 */
@DefaultProvider(DiffusionModelBuilder.class)
@Shareable
public class DiffusionModel implements Serializable {
    private static final long serialVersionUID = 1L;
    private final RealMatrix diffusionMatrix;

    public DiffusionModel(RealMatrix diffusionMatrix) {
        this.diffusionMatrix = diffusionMatrix;
    }

    /**
     * Get the diffusion matrix.
     * @return The Diffusion Matrix.
     */
    public RealMatrix getDiffusionMatrix() {
        return this.diffusionMatrix;
    }

}
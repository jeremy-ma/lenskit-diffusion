package org.grouplens.lenskit.hello;

import org.apache.commons.math3.linear.RealMatrix;
import org.grouplens.grapht.annotation.DefaultImplementation;
import org.grouplens.lenskit.vectors.similarity.CosineVectorSimilarity;

import java.util.HashMap;
import java.util.Map;

/**
 * Created by jeremyma on 11/08/15.
 */

@DefaultImplementation(ItemUtilityMatrixNormalizer.class)
public interface UtilityMatrixNormalizer {

    //normalize the utility matrix
    public RealMatrix normalizeUtilityMatrix(RealMatrix utility);
}

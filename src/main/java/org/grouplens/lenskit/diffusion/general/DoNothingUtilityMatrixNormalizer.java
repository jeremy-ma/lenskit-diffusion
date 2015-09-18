package org.grouplens.lenskit.diffusion.general;

import org.apache.commons.math3.linear.RealMatrix;

/**
 * Created by jeremyma on 17/09/15.
 */
public class DoNothingUtilityMatrixNormalizer implements UtilityMatrixNormalizer {

    public DoNothingUtilityMatrixNormalizer(){

    };

    @Override
    public RealMatrix normalize(RealMatrix utility) {
        System.out.println("Doing nothing");
        return utility;
    }
}

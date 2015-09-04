package org.grouplens.lenskit.diffusion.general;

/**
 * Created by jeremyma on 13/08/15.
 */

import org.grouplens.lenskit.core.Parameter;

import javax.inject.Qualifier;
import java.lang.annotation.*;

/**
 * fraction of nonzero values to remain in similarity matrix
 */
@Documented
@Parameter(Double.class)
@Qualifier
@Target({ElementType.METHOD, ElementType.PARAMETER})
@Retention(RetentionPolicy.RUNTIME)
public @interface ThresholdFraction {
}

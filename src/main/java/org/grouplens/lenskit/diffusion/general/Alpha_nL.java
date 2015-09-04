package org.grouplens.lenskit.diffusion.general;

/**
 * Created by jeremyma on 8/07/15.
 */

import org.grouplens.lenskit.core.Parameter;

import javax.inject.Qualifier;
import java.lang.annotation.*;


/**
 * value of alpha for the normalised laplacian
 */
@Documented
@Parameter(Double.class)
@Qualifier
@Target({ElementType.METHOD, ElementType.PARAMETER})
@Retention(RetentionPolicy.RUNTIME)
public @interface Alpha_nL {
}
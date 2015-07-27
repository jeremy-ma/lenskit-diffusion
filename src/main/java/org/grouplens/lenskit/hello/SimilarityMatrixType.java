package org.grouplens.lenskit.hello;

import org.grouplens.lenskit.core.Parameter;

import javax.inject.Qualifier;
import java.lang.annotation.*;


/**
 * String indicating filename for diffusion matrix
 */
@Documented
@Parameter(String.class)
@Qualifier
@Target({ElementType.METHOD, ElementType.PARAMETER})
@Retention(RetentionPolicy.RUNTIME)
public @interface SimilarityMatrixType {
}

package org.grouplens.lenskit.hello;

import org.grouplens.grapht.annotation.DefaultInteger;
import org.grouplens.lenskit.core.Parameter;

import javax.inject.Qualifier;


import java.lang.annotation.Documented;
import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;


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

package org.grouplens.lenskit.hello;

/**
 * Created by jeremyma on 8/07/15.
 */
//Alpha parameter for

import org.grouplens.lenskit.core.Parameter;

import javax.inject.Qualifier;
import java.lang.annotation.*;


/**
 * String indicating filename for diffusion matrix
 */
@Documented
@Parameter(Double.class)
@Qualifier
@Target({ElementType.METHOD, ElementType.PARAMETER})
@Retention(RetentionPolicy.RUNTIME)
public @interface Alpha {
}
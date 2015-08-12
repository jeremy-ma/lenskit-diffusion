package org.grouplens.lenskit.hello;

import org.apache.commons.math3.linear.RealVector;

import java.util.Iterator;

/**
 * Created by jeremyma on 12/08/15.
 */
public class VectorUtils {


    public static int countNonZero(RealVector v){
        int numNonZero = 0;
        Iterator itr = v.iterator();
        while(itr.hasNext()){
            itr.next();
            numNonZero++;
        }
        return numNonZero;
    }
}

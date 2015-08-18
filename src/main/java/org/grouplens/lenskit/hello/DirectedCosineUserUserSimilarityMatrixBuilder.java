package org.grouplens.lenskit.hello;

import org.apache.commons.math3.linear.RealMatrix;

import javax.inject.Inject;

/**
 * Created by jeremyma on 13/08/15.
 */
public class DirectedCosineUserUserSimilarityMatrixBuilder implements UserUserSimilarityMatrixBuilder {

    @Inject
    public DirectedCosineUserUserSimilarityMatrixBuilder(){

    }

    @Override
    public RealMatrix build(RealMatrix utility) {
        System.out.println("Making Directed Cosine UserUser similarity matrix");
        UserUserSimilarityMatrixBuilder directedBuilder = new DirectedUserUserSimilarityMatrixBuilder();
        UserUserSimilarityMatrixBuilder cosineBuilder = new CosineUserUserSimilarityMatrixBuilder();
        RealMatrix directed = directedBuilder.build(utility);
        RealMatrix sim = cosineBuilder.build(utility);
        int numUsers = utility.getRowDimension();

        //System.out.println(directed.getRowDimension());
        //pointwise multiplication
        for (int i=0; i<numUsers; i++){
            for (int j=0; j<numUsers; j++){
                //System.out.print(Integer.toString(i) + " " + Integer.toString(j) + "\n");
                sim.multiplyEntry(i,j,directed.getEntry(i,j));
            }
        }

        return sim;
    }

}

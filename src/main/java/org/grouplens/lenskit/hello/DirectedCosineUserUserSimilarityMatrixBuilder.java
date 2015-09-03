package org.grouplens.lenskit.hello;

import org.apache.commons.math3.linear.RealMatrix;

import javax.inject.Inject;
import java.util.ArrayList;

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
        //RealMatrix directed = directedBuilder.build(utility);
        RealMatrix sim = cosineBuilder.build(utility);
        int numUsers = utility.getRowDimension();
        int numItems = utility.getColumnDimension();

        //System.out.println(directed.getRowDimension());
        //pointwise multiplication
        ArrayList<Double> numRatedByUser = new ArrayList<Double>(numUsers);

        //count the number of ratings per user
        for (int i=0; i<numUsers; i++){
            double num = 0;
            for (int j=0; j<numItems; j++){
                if (utility.getEntry(i,j) > 0.0){
                    num += 1.0;
                }
            }
            numRatedByUser.add(i,num);
        }


        for (int i=0; i<numUsers; i++){
            for (int j=0; j<numUsers; j++){
                //System.out.print(Integer.toString(i) + " " + Integer.toString(j) + "\n");
                sim.multiplyEntry(i,j,1.0/numRatedByUser.get(i));
            }
        }

        return sim;
    }

}

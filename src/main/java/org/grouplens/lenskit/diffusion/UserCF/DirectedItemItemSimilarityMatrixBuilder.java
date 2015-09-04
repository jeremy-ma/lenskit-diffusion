package org.grouplens.lenskit.diffusion.UserCF;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

import javax.inject.Inject;
import java.util.HashMap;
import java.util.HashSet;

/**
 * Created by jeremyma on 13/08/15.
 */
public class DirectedItemItemSimilarityMatrixBuilder implements ItemItemSimilarityMatrixBuilder {

    @Inject
    public DirectedItemItemSimilarityMatrixBuilder(){}

    @Override
    public RealMatrix build(RealMatrix utility) {
        System.out.println("Making Directed similarity");

        HashMap<Integer, HashSet<Integer>> user_to_movies_watched = new HashMap<Integer, HashSet<Integer>>();

        int numUsers = utility.getRowDimension();
        int numItems = utility.getColumnDimension();

        RealMatrix similarity = MatrixUtils.createRealMatrix(numUsers, numUsers);

        //populate the user sets
        for (int i =0; i<numUsers;i++){
            if (! user_to_movies_watched.containsKey(i) ){
                user_to_movies_watched.put(i,new HashSet<Integer>());
            }
            for (int j=0; j<numItems;j++){
                if (utility.getEntry(i,j) > 0.0){
                    user_to_movies_watched.get(i).add(j);
                }
            }
        }

        // A(i,j) = (i and j) / i
        // number of movies watched by both i and j / number of movies watched by j

        for (int i=0; i<numUsers; i++){
            for (int j=0; j<numUsers; j++){
                HashSet<Integer> ui = new HashSet<Integer>(user_to_movies_watched.get(i));
                HashSet<Integer> uj = new HashSet<Integer>(user_to_movies_watched.get(j));
                uj.retainAll(ui); //calculate intersection
                double value = (double) uj.size() / (double) ui.size();
                similarity.setEntry(i,j,value);
            }
        }

        return similarity;
    }
}

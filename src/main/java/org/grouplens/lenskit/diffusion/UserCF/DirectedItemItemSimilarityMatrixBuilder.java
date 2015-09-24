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

        HashMap<Integer, HashSet<Integer>> items_to_users_watched = new HashMap<Integer, HashSet<Integer>>();

        int numUsers = utility.getRowDimension();
        int numItems = utility.getColumnDimension();

        RealMatrix similarity = MatrixUtils.createRealMatrix(numItems, numItems);

        //populate the item sets
        for (int i =0; i<numItems;i++){
            if (! items_to_users_watched.containsKey(i) ){
                items_to_users_watched.put(i, new HashSet<Integer>());
            }
            for (int j=0; j<numUsers;j++){
                if (utility.getEntry(j,i) != 0.0){
                    items_to_users_watched.get(i).add(j);
                }
            }
        }

        // A(i,j) = (i and j) / i
        // number of users watching both i and j / number of users who watched j

        for (int i=0; i<numItems; i++){
            for (int j=0; j<numItems; j++){
                HashSet<Integer> ui = new HashSet<Integer>(items_to_users_watched.get(i));
                HashSet<Integer> uj = new HashSet<Integer>(items_to_users_watched.get(j));
                uj.retainAll(ui); //calculate intersection
                double value = (double) uj.size() / (double) ui.size();
                similarity.setEntry(i,j,value);
            }
        }

        return similarity;
    }
}

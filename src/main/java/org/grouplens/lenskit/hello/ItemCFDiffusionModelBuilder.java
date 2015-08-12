package org.grouplens.lenskit.hello;

import com.jmatio.types.MLArray;
import com.jmatio.types.MLDouble;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.grouplens.lenskit.core.Transient;
import org.grouplens.lenskit.cursors.Cursor;
import org.grouplens.lenskit.data.dao.EventDAO;
import org.grouplens.lenskit.data.event.Rating;

import javax.inject.Inject;
import javax.inject.Provider;
import java.util.ArrayList;
import java.util.HashMap;

/**
 * Model builder that computes the global and item biases.
 */
public class ItemCFDiffusionModelBuilder implements Provider<DiffusionModel> {
    private RealMatrix diffMatrix=null;

    @Inject
    public ItemCFDiffusionModelBuilder(@Transient EventDAO dao, @Alpha_nL double alpha_nl, UtilityMatrixNormalizer normalizer,
                                       UserUserSimilarityMatrixBuilder similarityBuilder, LaplacianMatrixBuilder laplacianBuilder) {

        final HashMap<Long, HashMap<Long,Double>> ratingStore = new HashMap<Long, HashMap<Long,Double>>();
        long maxUserId = 943;
        long maxItemId = 1682;
        RealMatrix utility;
        RealMatrix similarity;
        //fill the utility matrix
        Cursor<Rating> ratings = dao.streamEvents(Rating.class);
        try {
            /* We loop over all ratings.  The 'fast()' improves performance for the common case,
             * when we will only work with the rating object inside the loop body.
             *
             * If the data set may have multiple ratings for the same (user,item) pair, this code
             * will be not quite correct.
             */

            for (Rating r: ratings.fast()) {
                //System.out.print("   ");
                if (!ratingStore.containsKey(r.getUserId())){
                    ratingStore.put(r.getUserId(), new HashMap<Long, Double>());
                }
                ratingStore.get(r.getUserId()).put(r.getItemId(),r.getValue());
            }
        } finally {
            // cursors must be closed
            ratings.close();
        }

        System.out.println("Creating Utility Matrix\n");
        utility = this.createUtilityMatrix((int)maxUserId, (int) maxItemId, ratingStore);
        System.out.println("Utility matrix created\n");
        //create a user-user similarity matrix (adjusted cosine)
        similarity = similarityBuilder.build(utility);
        System.out.println("Similarity Matrix create\n");
        //create a diffusion matrix (premultiplied by appropriate alpha)
        RealMatrix L = laplacianBuilder.build(similarity, alpha_nl);
        diffMatrix = MatrixUtils.createRealIdentityMatrix((int) maxUserId);
        diffMatrix = diffMatrix.add(L);
        diffMatrix = MatrixUtils.inverse(diffMatrix);

    }


    private RealMatrix createUtilityMatrix(int numUsers, int numItems,final HashMap<Long, HashMap<Long,Double>> ratingStore){
        System.out.println(numUsers);
        RealMatrix utility = MatrixUtils.createRealMatrix(numUsers, numItems);
        System.out.println("Transferring ratings");

        for (int user=0; user<numUsers; user++){
            for (int item=0; item<numItems; item++){
                if (ratingStore.containsKey((long)user+1) && ratingStore.get((long)user+1).containsKey((long) item + 1)) {
                    //System.out.println(ratingStore.get((long)user+1).get((long)item+1));
                    utility.setEntry(user,item,ratingStore.get((long)user+1).get((long) item + 1));
                } else {
                    utility.setEntry(user,item,0.0);
                }
            }

        }

        /*
        MLDouble MLutility = new MLDouble("utility", utility.getData());
        ArrayList<MLArray> collection = new ArrayList<MLArray>();
        collection.add(MLutility);
        */

        return utility;

    }

    @Override
    public DiffusionModel get(){
        return new DiffusionModel(diffMatrix);
    }
}

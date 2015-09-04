package org.grouplens.lenskit.diffusion.ItemCF;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.grouplens.lenskit.core.Transient;
import org.grouplens.lenskit.cursors.Cursor;
import org.grouplens.lenskit.data.dao.EventDAO;
import org.grouplens.lenskit.data.event.Rating;
import org.grouplens.lenskit.diffusion.general.*;

import javax.inject.Inject;
import javax.inject.Provider;
import java.util.HashMap;

/**
 * Model builder that computes the global and item biases.
 */
public class ItemCFDiffusionModelBuilder implements Provider<DiffusionModel> {
    private RealMatrix diffMatrix=null;

    @Inject
    public ItemCFDiffusionModelBuilder(@Transient EventDAO dao, @Alpha_nL double alpha_nl,
                                       @ThresholdFraction double thresholdFraction, UtilityMatrixNormalizer normalizer,
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

        utility = VectorUtils.createUtilityMatrix((int) maxUserId, (int) maxItemId, ratingStore);
        System.out.println("Utility matrix created");
        //create a user-user similarity matrix (adjusted cosine)
        similarity = similarityBuilder.build(utility);
        System.out.println("Similarity Matrix created");
        similarity = VectorUtils.thresholdSimilarityMatrix(similarity, thresholdFraction);
        VectorUtils.saveToFile(similarity, "afterThreshold.mat");

        //create a diffusion matrix (premultiplied by appropriate alpha)
        RealMatrix L = laplacianBuilder.build(similarity, alpha_nl);
        diffMatrix = MatrixUtils.createRealIdentityMatrix((int) maxUserId);
        diffMatrix = diffMatrix.add(L);
        diffMatrix = MatrixUtils.inverse(diffMatrix);
        VectorUtils.saveToFile(diffMatrix, "matt.mat");


    }


    @Override
    public DiffusionModel get(){
        return new DiffusionModel(diffMatrix);
    }
}

package org.grouplens.lenskit.diffusion.UserCF;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.grouplens.lenskit.core.Transient;
import org.grouplens.lenskit.cursors.Cursor;
import org.grouplens.lenskit.data.dao.EventDAO;
import org.grouplens.lenskit.data.event.Rating;
import org.grouplens.lenskit.diffusion.UserCF.ItemItemSimilarityMatrixBuilder;
import org.grouplens.lenskit.diffusion.general.*;

import javax.inject.Inject;
import javax.inject.Provider;
import java.util.HashMap;

/**
 * Model builder that computes the global and item biases.
 */
public class UserCFDiffusionModel implements DiffusionModel {
    private RealMatrix diffMatrix=null;

    @Inject
    public UserCFDiffusionModel(@Transient EventDAO dao, @Alpha_nL double alpha_nl,
                                @ThresholdFraction double thresholdFraction, UtilityMatrixNormalizer normalizer,
                                ItemItemSimilarityMatrixBuilder similarityBuilder, LaplacianMatrixBuilder laplacianBuilder) {

        final HashMap<Long, HashMap<Long,Double>> ratingStore = new HashMap<Long, HashMap<Long,Double>>();
        long maxUserId = 943;
        long maxItemId = 1682;
        RealMatrix utility;
        RealMatrix similarity;
        //fill the utility matrix
        Cursor<Rating> ratings = dao.streamEvents(Rating.class);
        try {
            for (Rating r: ratings.fast()) {
                //System.out.print("   ");
                if (!ratingStore.containsKey(r.getUserId())){
                    ratingStore.put(r.getUserId(), new HashMap<Long, Double>());
                }
                ratingStore.get(r.getUserId()).put(r.getItemId(),r.getValue());
            }
        } finally {
            ratings.close();
        }

        utility = VectorUtils.createUtilityMatrix((int) maxUserId, (int) maxItemId, ratingStore);
        System.out.println("Utility matrix created... normalizing");
        //create a user-user similarity matrix (adjusted cosine)
        utility = normalizer.normalize(utility);
        similarity = similarityBuilder.build(utility);
        System.out.println("Similarity Matrix created");
        similarity = VectorUtils.thresholdSimilarityMatrix(similarity, thresholdFraction);
        VectorUtils.saveToFile(similarity, "afterThreshold.mat");

        //create a diffusion matrix (premultiplied by appropriate alpha)
        RealMatrix L = laplacianBuilder.build(similarity, alpha_nl);
        diffMatrix = MatrixUtils.createRealIdentityMatrix((int) maxItemId);
        diffMatrix = diffMatrix.add(L);
        diffMatrix = MatrixUtils.inverse(diffMatrix);


        //VectorUtils.saveToFile(diffMatrix, "matt.mat");

    }

    /**
     * Get the diffusion matrix.
     *
     * @return The Diffusion Matrix.
     */
    @Override
    public RealMatrix getDiffusionMatrix() {
        return diffMatrix;
    }
}

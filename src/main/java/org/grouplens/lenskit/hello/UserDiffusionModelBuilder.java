package org.grouplens.lenskit.hello;

import com.jmatio.io.MatFileWriter;
import com.jmatio.types.MLArray;
import com.jmatio.types.MLDouble;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
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
public class UserDiffusionModelBuilder implements Provider<DiffusionModel> {
    private final EventDAO dao;
    private RealMatrix diffMatrix=null;

    @Inject
    public UserDiffusionModelBuilder(@Transient EventDAO dao, @Alpha_nL double alpha) {
        this.dao = dao;
        final HashMap<Long, HashMap<Long,Double>> ratingStore = new HashMap<Long, HashMap<Long,Double>>();
        long maxUserId = 943;
        long maxItemId = 1682;
        RealMatrix utility;
        RealMatrix similarity;
        //fill the utility matrix

        System.out.println("Boom");
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
        utility = createUtilityMatrix((int)maxUserId, (int) maxItemId, ratingStore);
        System.out.println("Utility matrix created\n");
        //create a user-user similarity matrix (adjusted cosine)
        similarity = createSimilarityMatrix((int)maxUserId, utility);
        System.out.println("Similarity Matrix create\n");
        //create a normalized diffusion matrix
        RealMatrix L_n = getNormalizedLaplacian(similarity);
        L_n = L_n.scalarMultiply(alpha);
        diffMatrix = MatrixUtils.createRealIdentityMatrix((int) maxUserId);
        diffMatrix = diffMatrix.add(L_n);
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


        MLDouble MLutility = new MLDouble("utility", utility.getData());
        ArrayList<MLArray> collection = new ArrayList<MLArray>();
        collection.add(MLutility);

        try {
            System.out.println("Saving utility matrix");
            MatFileWriter writer = new MatFileWriter("utility.mat", collection);
        } catch (Exception e) {
            System.out.println("failed to save utility");
        }

        return utility;

    }

    private RealMatrix createSimilarityMatrix(int numUsers, RealMatrix utility){
        RealMatrix similarity = MatrixUtils.createRealMatrix(numUsers,numUsers);

        // create adjusted cosine similarity matrix
        for (int i=0; i<numUsers; i++){
            for (int j=i; j<numUsers; j++){
                double simVal = 0.0;
                if (utility.getRowVector(i).getNorm() != 0 && utility.getRowVector(j).getNorm() != 0) {
                    simVal = utility.getRowVector(i).cosine(utility.getRowVector(j));
                }
                similarity.setEntry(i,j,simVal);
                similarity.setEntry(j,i,simVal);
            }
        }

        return similarity;
    }

    private RealMatrix getNormalizedLaplacian(RealMatrix similarity){
        RealMatrix laplacian = similarity.copy();
        for (int i=0; i<similarity.getRowDimension(); i++){
            RealVector v = laplacian.getRowVector(i);
            double degree = v.getL1Norm(); //actually Dii + 1.0 but cancels in next step
            v.mapMultiplyToSelf(-1.0);
            v.addToEntry(i, degree);
            //normalize
            v.mapDivideToSelf(degree);
            laplacian.setRowVector(i,v);
        }
        return laplacian;
    }

    @Override
    public DiffusionModel get() {


        return new DiffusionModel(diffMatrix);
    }
}

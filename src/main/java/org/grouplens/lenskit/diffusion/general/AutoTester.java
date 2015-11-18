package org.grouplens.lenskit.diffusion.general;

import org.grouplens.lenskit.ItemScorer;
import org.grouplens.lenskit.baseline.*;
import org.grouplens.lenskit.core.LenskitConfiguration;
import org.grouplens.lenskit.diffusion.ItemCF.*;
import org.grouplens.lenskit.diffusion.Iterative.IterativeDiffusionItemScorer;
import org.grouplens.lenskit.diffusion.UserCF.*;
import org.grouplens.lenskit.diffusion.vectorsimilarity.DiffusedCosineVectorSimilarity;
import org.grouplens.lenskit.eval.EvalConfig;
import org.grouplens.lenskit.eval.algorithm.AlgorithmInstance;
import org.grouplens.lenskit.eval.data.CSVDataSource;
import org.grouplens.lenskit.eval.data.CSVDataSourceBuilder;
import org.grouplens.lenskit.eval.metrics.predict.CoveragePredictMetric;
import org.grouplens.lenskit.eval.metrics.predict.MAEPredictMetric;
import org.grouplens.lenskit.eval.metrics.predict.NDCGPredictMetric;
import org.grouplens.lenskit.eval.metrics.predict.RMSEPredictMetric;
import org.grouplens.lenskit.eval.traintest.CachingDAOProvider;
import org.grouplens.lenskit.eval.traintest.SimpleEvaluator;
import org.grouplens.lenskit.knn.NeighborhoodSize;
import org.grouplens.lenskit.knn.item.ItemItemScorer;
import org.grouplens.lenskit.knn.user.NeighborFinder;
import org.grouplens.lenskit.knn.user.SnapshotNeighborFinder;
import org.grouplens.lenskit.knn.user.UserUserItemScorer;
import org.grouplens.lenskit.transform.normalize.BaselineSubtractingUserVectorNormalizer;
import org.grouplens.lenskit.transform.normalize.UserVectorNormalizer;
import org.grouplens.lenskit.vectors.similarity.CosineVectorSimilarity;
import org.grouplens.lenskit.vectors.similarity.VectorSimilarity;

import java.io.File;
import java.util.Properties;

/**
 * Created by jeremyma on 26/09/15.
 */
public class AutoTester {



    public static void main(String[] args) {
        itemCFEval();
        userCFEval();
        //iterativeDiffusionEval();

    }

    /*
    itemCF all tests
     */
    private static void itemCFEval(){
        Class similaritytypes [] = {CosineUserUserSimilarityMatrixBuilder.class, CosineUserUserSimilarityMatrixBuilder.class,
                DirectedUserUserSimilarityMatrixBuilder.class, PearsonCorrelationUserUserSimilarityMatrixBuilder.class};
        Class normalizers [] = {DoNothingUtilityMatrixNormalizer.class, UserUtilityMatrixNormalizer.class,
                                DoNothingUtilityMatrixNormalizer.class, DoNothingUtilityMatrixNormalizer.class};
        Class laplacians [] = {/*NormalizedLaplacianMatrixBuilder.class,*/ RegularLaplacianMatrixBuilder.class};
        int neighbourhoodsizes []= {30};
        double meanDamping = 5.0;
        /*
        parameters for normalised diffusion
         */
        double normalized_alphas [] = {1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0};
        double normalized_thresholds [] = {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8};

        /*
        parameters for regular diffusion
         */
        double regular_alphas [] = {1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0};
        //double regular_alphas [] = {13.0,14.0,15.0,16.0,};
        double regular_thresholds [] = {1.0};

        double alphas [];
        double thresholds [];

        ParameterSearch search = null;

        for (int i=0; i < similaritytypes.length; i++){
            for (Class laplaciantype:laplacians){
                search = new ParameterSearch(ItemItemScorer.class, UserMeanItemScorer.class,
                        ItemMeanRatingItemScorer.class, BaselineSubtractingUserVectorNormalizer.class,
                        SnapshotNeighborFinder.class, ItemCFDiffusionModel.class, similaritytypes[i],
                        normalizers[i], laplaciantype, DiffusedCosineVectorSimilarity.class, CosineVectorSimilarity.class);
                if (laplaciantype == NormalizedLaplacianMatrixBuilder.class){
                    alphas= normalized_alphas;
                    thresholds = normalized_thresholds;
                } else {
                    alphas = regular_alphas;
                    thresholds = regular_thresholds;
                }
                System.out.println("ItemItemScorer: "+similaritytypes[i].toString()+" "+normalizers[i].toString());
                search.runWithParameters(neighbourhoodsizes, alphas,thresholds,null);

            }
        }

        search.runRegular(30,new Double(0.8));

    }


  /*
userCF all tests
 */
    private static void userCFEval(){

        Class similaritytypes [] = {/*CosineItemItemSimilarityMatrixBuilder.class, CosineItemItemSimilarityMatrixBuilder.class,
                DirectedItemItemSimilarityMatrixBuilder.class,*/ PearsonCorrelationItemItemSimilarityMatrixBuilder.class};
        Class normalizers [] = {/*DoNothingUtilityMatrixNormalizer.class, UserUtilityMatrixNormalizer.class,
                DoNothingUtilityMatrixNormalizer.class,*/ DoNothingUtilityMatrixNormalizer.class};



        //Class similaritytypes [] = {/*CosineItemItemSimilarityMatrixBuilder.class,*/PearsonCorrelationItemItemSimilarityMatrixBuilder.class};
        //Class normalizers [] = {/*UserUtilityMatrixNormalizer.class,*/ DoNothingUtilityMatrixNormalizer.class};

        Class laplacians [] = {/*NormalizedLaplacianMatrixBuilder.class, */RegularLaplacianMatrixBuilder.class};
        int neighbourhoodsizes []= {30};
        double meanDamping = 5.0;
        /*
        parameters for normalised diffusion
         */
        //double normalized_alphas [] = {1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0};
        //double normalized_thresholds [] = {0.05,0.075,0.1,0.125,0.15};
        double normalized_alphas [] = {4.0};
        double normalized_thresholds [] = {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0};
        /*
        parameters for regular diffusion
         */
        //double regular_alphas [] = {1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0};
        double regular_alphas [] = {1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0};
        double regular_thresholds [] = {0.1};

        double alphas [];
        double thresholds [];

        ParameterSearch search = null;


        for (int i=0; i < similaritytypes.length; i++){
            for (Class laplaciantype:laplacians){
                search = new ParameterSearch(UserUserItemScorer.class, UserMeanItemScorer.class,
                        ItemMeanRatingItemScorer.class, BaselineSubtractingUserVectorNormalizer.class,
                        SnapshotNeighborFinder.class, UserCFDiffusionModel.class, similaritytypes[i],
                        normalizers[i], laplaciantype, DiffusedCosineVectorSimilarity.class, CosineVectorSimilarity.class);
                if (laplaciantype == NormalizedLaplacianMatrixBuilder.class){
                    alphas= normalized_alphas;
                    thresholds = normalized_thresholds;
                } else {
                    alphas = regular_alphas;
                    thresholds = regular_thresholds;
                }
                System.out.println("UserUserScorer: "+similaritytypes[i].toString()+" "+normalizers[i].toString());
                search.runWithParameters(neighbourhoodsizes, alphas,thresholds,new Double(0.8));

            }
        }

        search.runRegular(30,new Double(0.8));

    }

    /*
    IterativeDiffusion
     */
    private static void iterativeDiffusionEval(){
        LenskitConfiguration config = new LenskitConfiguration();
        config.bind(ItemScorer.class).to(IterativeDiffusionItemScorer.class);
        config.bind(BaselineScorer.class, ItemScorer.class).to(UserMeanItemScorer.class);
        config.bind(UserMeanBaseline.class, ItemScorer.class).to(ItemMeanRatingItemScorer.class);
        config.bind(UserVectorNormalizer.class).to(BaselineSubtractingUserVectorNormalizer.class);
        config.bind(ItemItemSimilarityMatrixBuilder.class).to(PearsonCorrelationItemItemSimilarityMatrixBuilder.class);
        config.bind(UtilityMatrixNormalizer.class).to(DoNothingUtilityMatrixNormalizer.class);
        config.bind(LaplacianMatrixBuilder.class).to(NormalizedLaplacianMatrixBuilder.class);
        config.set(Alpha_nL.class).to(1.0);
        config.set(ThresholdFraction.class).to(0.1);



        AlgorithmInstance algo = new AlgorithmInstance("alg",config);

        Properties EvalProps = new Properties();
        EvalProps.setProperty(EvalConfig.THREAD_COUNT_PROPERTY, Integer.toString(5));
        SimpleEvaluator simpleEval = new SimpleEvaluator(EvalProps);
        simpleEval.addAlgorithm(algo);

        File in = new File("ml-100k/u.data");
        CSVDataSourceBuilder builder = new CSVDataSourceBuilder(in);
        builder.setDelimiter("\t");
        CSVDataSource dat = builder.build();

        //use 5-fold CV
        //use the default 10 ratings from each user for test set
        simpleEval.addDataset(dat,5);

        //Add metrics
        RMSEPredictMetric rmse = new RMSEPredictMetric();
        CoveragePredictMetric cover = new CoveragePredictMetric();
        NDCGPredictMetric ndcg = new NDCGPredictMetric();
        MAEPredictMetric mae = new MAEPredictMetric();

        simpleEval.addMetric(rmse);
        simpleEval.addMetric(cover);
        simpleEval.addMetric(ndcg);
        simpleEval.addMetric(mae);

        /*
        Save in nested file hierarchy
         */
        File out = new File("iterativeresults"+".csv");
        simpleEval.setOutput(out);
        try{
            simpleEval.call();
        } catch (Exception e){
            System.out.println(e.getMessage());
            e.printStackTrace();
        }
    }

}

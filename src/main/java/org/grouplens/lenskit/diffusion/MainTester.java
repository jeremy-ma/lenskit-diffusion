/*
 * Copyright 2011-2014 LensKit Contributors.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
package org.grouplens.lenskit.diffusion;

import org.apache.commons.lang3.builder.Diff;
import org.grouplens.lenskit.ItemScorer;
import org.grouplens.lenskit.baseline.*;
import org.grouplens.lenskit.core.LenskitConfiguration;
import org.grouplens.lenskit.data.pref.PreferenceDomain;
import org.grouplens.lenskit.data.pref.PreferenceDomainBuilder;
import org.grouplens.lenskit.diffusion.ItemCF.*;
import org.grouplens.lenskit.diffusion.UserCF.*;
import org.grouplens.lenskit.diffusion.general.*;
import org.grouplens.lenskit.diffusion.vectorsimilarity.DiffusedCosineVectorSimilarity;
import org.grouplens.lenskit.diffusion.org.grouplens.lenskit.diffusion.unused.DiffusedDistanceVectorSimilarity;
import org.grouplens.lenskit.diffusion.vectorsimilarity.DiffusedPearsonCorrelation;
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
import org.grouplens.lenskit.diffusion.org.grouplens.lenskit.diffusion.unused.DoubleDiffusionItemScorer;
import org.grouplens.lenskit.diffusion.org.grouplens.lenskit.diffusion.unused.PrediffusedItemCosineSimilarity;
import org.grouplens.lenskit.diffusion.org.grouplens.lenskit.diffusion.unused.PrediffusedUserCosineSimilarity;
import org.grouplens.lenskit.iterative.IterationCount;
import org.grouplens.lenskit.iterative.RegularizationTerm;
import org.grouplens.lenskit.knn.NeighborhoodSize;
import org.grouplens.lenskit.knn.item.ItemItemScorer;
import org.grouplens.lenskit.knn.item.ItemSimilarity;
import org.grouplens.lenskit.knn.item.WeightedAverageNeighborhoodScorer;
import org.grouplens.lenskit.knn.user.NeighborFinder;
import org.grouplens.lenskit.knn.user.SnapshotNeighborFinder;
import org.grouplens.lenskit.knn.user.UserSimilarity;
import org.grouplens.lenskit.knn.user.UserUserItemScorer;
import org.grouplens.lenskit.mf.funksvd.FeatureCount;
import org.grouplens.lenskit.mf.funksvd.FunkSVDItemScorer;
import org.grouplens.lenskit.slopeone.DeviationDamping;
import org.grouplens.lenskit.slopeone.SlopeOneItemScorer;
import org.grouplens.lenskit.slopeone.WeightedSlopeOneItemScorer;
import org.grouplens.lenskit.transform.normalize.BaselineSubtractingUserVectorNormalizer;
import org.grouplens.lenskit.transform.normalize.UserVectorNormalizer;
import org.grouplens.lenskit.vectors.similarity.*;

import javax.xml.ws.Provider;
import java.io.File;
import java.util.Properties;

/**
 * Demonstration app for LensKit. This application builds an item-item CF model
 * from a CSV file, then generates recommendations for a user.
 *
 * Usage: java org.grouplens.lenskit.diffusion.MainTester ratings.csv user
 */
public class MainTester implements Runnable {
    public static void main(String[] args) {
        MainTester hello = new MainTester(args);
        System.out.println("Hellooo");
        try {
            //runsingle();
            hello.run();
        } catch (RuntimeException e) {
            System.err.println(e.toString());
            e.printStackTrace(System.err);
            System.exit(1);
        }
    }

    private String dataFileName = "ml-100k/u.data";
    private String resultsFileName = "results.csv";
    private String vectorSimilarityMeasure = "cosine";
    private int numNeighbours = 30;

    private void set_config_userCF(LenskitConfiguration config){
        config.bind(ItemScorer.class)
                .to(UserUserItemScorer.class);
        config.bind(BaselineScorer.class, ItemScorer.class)
                .to(UserMeanItemScorer.class);
        config.bind(UserMeanBaseline.class, ItemScorer.class)
                .to(ItemMeanRatingItemScorer.class);
        config.bind(UserVectorNormalizer.class)
                .to(BaselineSubtractingUserVectorNormalizer.class);
        config.set(NeighborhoodSize.class).to(numNeighbours);
        config.set(MeanDamping.class).to(5.0);
        config.bind(NeighborFinder.class).to(SnapshotNeighborFinder.class);
    }

    private void set_config_itemCF(LenskitConfiguration config){
        config.bind(ItemScorer.class).to(ItemItemScorer.class);
        config.bind(BaselineScorer.class, ItemScorer.class)
                .to(UserMeanItemScorer.class);
        config.bind(UserMeanBaseline.class, ItemScorer.class)
                .to(ItemMeanRatingItemScorer.class);
        config.bind(UserVectorNormalizer.class)
                .to(BaselineSubtractingUserVectorNormalizer.class);
        config.set(NeighborhoodSize.class).to(numNeighbours);
        config.set(MeanDamping.class).to(5.0);
    }

    private void set_config_FunkSVD(LenskitConfiguration config){
        config.bind(ItemScorer.class).to(FunkSVDItemScorer.class);

        config.set(IterationCount.class).to(150); // defaults (slightly changed from kluver paper)
        config.set(MeanDamping.class).to(5.0);
        //config.set(FeatureCount.class).to(45);
        //config.set(RegularizationTerm.class).to(0.0005);

        /*
        config.set(IterationCount.class).to(itercount); // these settings were used in kluver paper
        config.set(MeanDamping.class).to(meanDamping);
        config.set(FeatureCount.class).to(featureCount);
        config.set(RegularizationTerm.class).to(regularizationTerm);
        */
        config.bind(BaselineScorer.class, ItemScorer.class)
                        .to(UserMeanItemScorer.class);
        config.bind(UserMeanBaseline.class, ItemScorer.class)
                .to(ItemMeanRatingItemScorer.class);
        config.bind(UserVectorNormalizer.class)
                .to(BaselineSubtractingUserVectorNormalizer.class);
    }

    private void set_config_mean_predictor(LenskitConfiguration config){
        config.bind(ItemScorer.class)
                .to(UserMeanItemScorer.class);
        config.bind(BaselineScorer.class, ItemScorer.class)
                .to(UserMeanItemScorer.class);
        config.bind(UserMeanBaseline.class, ItemScorer.class)
                .to(ItemMeanRatingItemScorer.class);
        config.bind(UserVectorNormalizer.class)
                .to(BaselineSubtractingUserVectorNormalizer.class);
        config.set(NeighborhoodSize.class).to(numNeighbours);
        config.bind(NeighborFinder.class).to(SnapshotNeighborFinder.class);
    }

    private void set_config_precomputed(LenskitConfiguration config){
        config.bind(ItemScorer.class).to(DoubleDiffusionItemScorer.class);
        config.bind(BaselineScorer.class, ItemScorer.class)
                .to(UserMeanItemScorer.class);
        config.bind(UserMeanBaseline.class, ItemScorer.class)
                .to(ItemMeanRatingItemScorer.class);
        config.bind(UserVectorNormalizer.class)
                .to(BaselineSubtractingUserVectorNormalizer.class);
    }

    private SimpleEvaluator UserUserEval(int numThreads){
        LenskitConfiguration config_reg = new LenskitConfiguration();
        set_config_userCF(config_reg);
        LenskitConfiguration config_diff = new LenskitConfiguration();
        set_config_userCF(config_diff);
        LenskitConfiguration config_diff_n = new LenskitConfiguration();
        set_config_userCF(config_diff_n);
        LenskitConfiguration config_diff_abs = new LenskitConfiguration();
        set_config_userCF(config_diff_abs);
        LenskitConfiguration config_diff_abs_n = new LenskitConfiguration();
        set_config_userCF(config_diff_abs_n);


        if (vectorSimilarityMeasure.equalsIgnoreCase("cosine")){
            config_reg.bind(VectorSimilarity.class).to(CosineVectorSimilarity.class);
            config_diff.bind(VectorSimilarity.class).to(DiffusedCosineVectorSimilarity.class);
            config_diff_n.bind(VectorSimilarity.class).to(DiffusedCosineVectorSimilarity.class);
            config_diff_abs.bind(VectorSimilarity.class).to(DiffusedCosineVectorSimilarity.class);
        } else if (vectorSimilarityMeasure.equalsIgnoreCase("pearson")){
            config_reg.bind(VectorSimilarity.class).to(PearsonCorrelation.class);
            config_diff.bind(VectorSimilarity.class).to(DiffusedPearsonCorrelation.class);
            config_diff_n.bind(VectorSimilarity.class).to(DiffusedPearsonCorrelation.class);
            config_diff_abs.bind(VectorSimilarity.class).to(DiffusedPearsonCorrelation.class);
        } else {
            config_reg.bind(VectorSimilarity.class).to(DistanceVectorSimilarity.class);
            config_diff.bind(VectorSimilarity.class).to(DiffusedDistanceVectorSimilarity.class);
            config_diff_n.bind(VectorSimilarity.class).to(DiffusedDistanceVectorSimilarity.class);
            config_diff_abs.bind(VectorSimilarity.class).to(DiffusedDistanceVectorSimilarity.class);
        }

        config_diff.set(MatrixFileName.class).to("ml100k_udiff.mat");
        config_diff_n.set(MatrixFileName.class).to("ml100k_udiff_n.mat");
        config_diff_abs.set(MatrixFileName.class).to("ml100k_udiff_abs.mat");
        config_diff_abs_n.set(MatrixFileName.class).to("ml100k_udiff_abs_n.mat");

        AlgorithmInstance regular_algo = new AlgorithmInstance("regular_" + vectorSimilarityMeasure + "_similarity", config_reg);
        AlgorithmInstance diffusion_algo = new AlgorithmInstance("diffusion_" + vectorSimilarityMeasure + "_similarity", config_diff);
        AlgorithmInstance diffusion_norm_algo = new AlgorithmInstance("diffusion_norm_" + vectorSimilarityMeasure + "_similarity", config_diff_n);
        AlgorithmInstance diffusion_abs_algo = new AlgorithmInstance("diffusion_abs_" + vectorSimilarityMeasure + "_similarity", config_diff_abs);
        AlgorithmInstance diffusion_abs_norm_algo = new AlgorithmInstance("diffusion_abs_norm_" + vectorSimilarityMeasure + "_similarity", config_diff_abs_n);

        //set to run with n threads
        Properties EvalProps = new Properties();
        EvalProps.setProperty(EvalConfig.THREAD_COUNT_PROPERTY, Integer.toString(numThreads));
        SimpleEvaluator simpleEval = new SimpleEvaluator(EvalProps);
        simpleEval.addAlgorithm(diffusion_algo);
        simpleEval.addAlgorithm(regular_algo);
        simpleEval.addAlgorithm(diffusion_norm_algo);
        simpleEval.addAlgorithm(diffusion_abs_algo);
        simpleEval.addAlgorithm(diffusion_abs_norm_algo);

        return simpleEval;
    }

    private SimpleEvaluator ItemItemEval(int numThreads){
        LenskitConfiguration config_diff_n = new LenskitConfiguration();
        LenskitConfiguration config_reg = new LenskitConfiguration();
        LenskitConfiguration config_diff = new LenskitConfiguration();

        set_config_itemCF(config_diff_n);
        set_config_itemCF(config_reg);
        set_config_itemCF(config_diff);

        config_diff_n.set(MatrixFileName.class).to("ml100k_udiff_n.mat");
        config_diff.set(MatrixFileName.class).to("ml100k_udiff.mat");

        if (vectorSimilarityMeasure.equalsIgnoreCase("cosine")){
            config_reg.bind(VectorSimilarity.class).to(CosineVectorSimilarity.class);
            config_diff_n.bind(VectorSimilarity.class).to(DiffusedCosineVectorSimilarity.class);
            config_diff.bind(VectorSimilarity.class).to(DiffusedCosineVectorSimilarity.class);
        } else {
            config_reg.bind(VectorSimilarity.class).to(PearsonCorrelation.class);
            config_diff_n.bind(VectorSimilarity.class).to(DiffusedPearsonCorrelation.class);
            config_diff.bind(VectorSimilarity.class).to(DiffusedPearsonCorrelation.class);
        }


        AlgorithmInstance regular_algo = new AlgorithmInstance("regular_" + vectorSimilarityMeasure + "_similarity_itemitemCF", config_reg);
        AlgorithmInstance diffusion_algo = new AlgorithmInstance("diffusion_" + vectorSimilarityMeasure + "_similarity_itemitemCF", config_diff);
        AlgorithmInstance diffusion_norm_algo = new AlgorithmInstance("diffusion_norm_" + vectorSimilarityMeasure + "_similarityitemitemCF", config_diff_n);

        //set to run with n threads
        Properties EvalProps = new Properties();
        EvalProps.setProperty(EvalConfig.THREAD_COUNT_PROPERTY, Integer.toString(numThreads));
        SimpleEvaluator simpleEval = new SimpleEvaluator(EvalProps);
        simpleEval.addAlgorithm(diffusion_algo);
        simpleEval.addAlgorithm(regular_algo);
        simpleEval.addAlgorithm(diffusion_norm_algo);

        return simpleEval;
    }

    private SimpleEvaluator DoubleDiffItemBasedCFEval(int numThreads){
        LenskitConfiguration config_diff_n = new LenskitConfiguration();
        LenskitConfiguration config_reg = new LenskitConfiguration();
        LenskitConfiguration config_diff = new LenskitConfiguration();
        LenskitConfiguration config_precomputed = new LenskitConfiguration();
        LenskitConfiguration config_precomputed_n = new LenskitConfiguration();
        LenskitConfiguration config_mean = new LenskitConfiguration();

        set_config_itemCF(config_diff_n);
        set_config_itemCF(config_reg);
        set_config_itemCF(config_diff);
        set_config_precomputed(config_precomputed);
        set_config_precomputed(config_precomputed_n);
        set_config_mean_predictor(config_mean);

        config_diff_n.set(MatrixFileName.class).to("ml100k_util_diff_n.mat");
        config_diff.set(MatrixFileName.class).to("ml100k_util_diff.mat");
        config_precomputed.set(MatrixFileName.class).to("ml100k_util_diff_complete.mat");
        config_precomputed_n.set(MatrixFileName.class).to("ml100k_util_diff_n_complete.mat");

        if (vectorSimilarityMeasure.equalsIgnoreCase("cosine")){
            config_reg.bind(VectorSimilarity.class).to(CosineVectorSimilarity.class);
            config_diff_n.bind(ItemSimilarity.class).to(PrediffusedItemCosineSimilarity.class);
            config_diff.bind(ItemSimilarity.class).to(PrediffusedItemCosineSimilarity.class);
        } else {
            config_reg.bind(VectorSimilarity.class).to(PearsonCorrelation.class);
            config_diff_n.bind(VectorSimilarity.class).to(DiffusedPearsonCorrelation.class);
            config_diff.bind(VectorSimilarity.class).to(DiffusedPearsonCorrelation.class);
        }

        AlgorithmInstance regular_algo = new AlgorithmInstance("regular_" + vectorSimilarityMeasure + "_similarity_itemitemCF", config_reg);
        AlgorithmInstance diffusion_algo = new AlgorithmInstance("doublediffusion_" + vectorSimilarityMeasure + "_similarity_itemitemCF", config_diff);
        AlgorithmInstance diffusion_norm_algo = new AlgorithmInstance("doublediffusion_norm_" + vectorSimilarityMeasure + "_similarityitemitemCF", config_diff_n);
        AlgorithmInstance diff_matrix_completion = new AlgorithmInstance("diffusion_matrix_completion",config_precomputed);
        AlgorithmInstance diff_n_matrix_completion = new AlgorithmInstance("diffusion_norm_matrix_completion",config_precomputed_n);
        AlgorithmInstance mean_algo = new AlgorithmInstance("mean_predictor", config_mean);

        //set to run with n threads
        Properties EvalProps = new Properties();
        EvalProps.setProperty(EvalConfig.THREAD_COUNT_PROPERTY, Integer.toString(numThreads));
        SimpleEvaluator simpleEval = new SimpleEvaluator(EvalProps);
        simpleEval.addAlgorithm(diffusion_algo);
        simpleEval.addAlgorithm(regular_algo);
        simpleEval.addAlgorithm(diffusion_norm_algo);
        simpleEval.addAlgorithm(diff_matrix_completion);
        simpleEval.addAlgorithm(diff_n_matrix_completion);
        //simpleEval.addAlgorithm(mean_algo);

        return simpleEval;
    }

    private SimpleEvaluator DoubleDiffUserBasedCFEval(int numThreads){
        LenskitConfiguration config_diff_n = new LenskitConfiguration();
        LenskitConfiguration config_reg = new LenskitConfiguration();
        LenskitConfiguration config_diff = new LenskitConfiguration();
        LenskitConfiguration config_precomputed = new LenskitConfiguration();
        LenskitConfiguration config_precomputed_n = new LenskitConfiguration();
        LenskitConfiguration config_mean = new LenskitConfiguration();

        set_config_userCF(config_diff_n);
        set_config_userCF(config_reg);
        set_config_userCF(config_diff);
        set_config_precomputed(config_precomputed);
        set_config_precomputed(config_precomputed_n);
        set_config_mean_predictor(config_mean);

        config_diff_n.set(MatrixFileName.class).to("ml100k_util_diff_n.mat");
        config_diff.set(MatrixFileName.class).to("ml100k_util_diff.mat");
        config_precomputed.set(MatrixFileName.class).to("ml100k_util_diff_complete.mat");
        config_precomputed_n.set(MatrixFileName.class).to("ml100k_util_diff_n_complete.mat");

        if (vectorSimilarityMeasure.equalsIgnoreCase("cosine")){
            config_reg.bind(VectorSimilarity.class).to(CosineVectorSimilarity.class);
            config_diff_n.bind(UserSimilarity.class).to(PrediffusedUserCosineSimilarity.class);
            config_diff.bind(UserSimilarity.class).to(PrediffusedUserCosineSimilarity.class);
        } else {
            config_reg.bind(VectorSimilarity.class).to(PearsonCorrelation.class);
            config_diff_n.bind(VectorSimilarity.class).to(DiffusedPearsonCorrelation.class);
            config_diff.bind(VectorSimilarity.class).to(DiffusedPearsonCorrelation.class);
        }

        AlgorithmInstance regular_algo = new AlgorithmInstance("regular_" + vectorSimilarityMeasure + "_similarity_UserCF", config_reg);
        AlgorithmInstance diffusion_algo = new AlgorithmInstance("doublediffusion_" + vectorSimilarityMeasure + "_similarity_UserCF", config_diff);
        AlgorithmInstance diffusion_norm_algo = new AlgorithmInstance("doublediffusion_norm_" + vectorSimilarityMeasure + "_similarity_UserCF", config_diff_n);
        AlgorithmInstance diff_matrix_completion = new AlgorithmInstance("diffusion_matrix_completion",config_precomputed);
        AlgorithmInstance diff_n_matrix_completion = new AlgorithmInstance("diffusion_norm_matrix_completion",config_precomputed_n);
        AlgorithmInstance mean_algo = new AlgorithmInstance("mean_predictor", config_mean);

        //set to run with n threads
        Properties EvalProps = new Properties();
        EvalProps.setProperty(EvalConfig.THREAD_COUNT_PROPERTY, Integer.toString(numThreads));
        SimpleEvaluator simpleEval = new SimpleEvaluator(EvalProps);
        simpleEval.addAlgorithm(diffusion_algo);
        simpleEval.addAlgorithm(regular_algo);
        simpleEval.addAlgorithm(diffusion_norm_algo);
        simpleEval.addAlgorithm(diff_matrix_completion);
        simpleEval.addAlgorithm(diff_n_matrix_completion);
        simpleEval.addAlgorithm(mean_algo);

        return simpleEval;
    }

    private SimpleEvaluator SVDEval(int numThreads,
                                    int featureCount, double regularizationTerm){
        LenskitConfiguration config_svd = new LenskitConfiguration();
        set_config_FunkSVD(config_svd);
        config_svd.set(RegularizationTerm.class).to(regularizationTerm);
        config_svd.set(FeatureCount.class).to(featureCount);

        AlgorithmInstance svd_alg = new AlgorithmInstance("SVD_iteration150_meandamping5_featurecount"+String.valueOf(featureCount), config_svd);
        Properties EvalProps = new Properties();
        EvalProps.setProperty(EvalConfig.THREAD_COUNT_PROPERTY, Integer.toString(numThreads));
        SimpleEvaluator simpleEval = new SimpleEvaluator(EvalProps);
        simpleEval.addAlgorithm(svd_alg);
        return simpleEval;
    }

    private SimpleEvaluator testEval1(int numThreads, double alpha, double thresholdFraction){
        /* itemUser normalization, itemCF, cosine similarity, cosine vector similarity */


        LenskitConfiguration config_diff_n = new LenskitConfiguration();
        LenskitConfiguration config_reg = new LenskitConfiguration();
        LenskitConfiguration config_diff = new LenskitConfiguration();

        set_config_itemCF(config_diff_n);
        set_config_itemCF(config_diff);
        set_config_itemCF(config_reg);

        //set normalized settings
        config_diff_n.set(Alpha_nL.class).to(alpha);
        config_diff_n.set(ThresholdFraction.class).to(thresholdFraction);
        config_diff_n.bind(DiffusionModel.class).to(ItemCFDiffusionModel.class);
        config_diff_n.bind(UserUserSimilarityMatrixBuilder.class).to(CosineUserUserSimilarityMatrixBuilder.class);
        config_diff_n.bind(UtilityMatrixNormalizer.class).to(ItemUserUtilityMatrixNormalizer.class);

        //set non-normalized settings
        config_diff.set(Alpha_nL.class).to(alpha);
        config_diff.set(ThresholdFraction.class).to(thresholdFraction);
        config_diff.bind(DiffusionModel.class).to(ItemCFDiffusionModel.class);
        config_diff.bind(UserUserSimilarityMatrixBuilder.class).to(CosineUserUserSimilarityMatrixBuilder.class);
        config_diff.bind(UtilityMatrixNormalizer.class).to(ItemUserUtilityMatrixNormalizer.class);
        config_diff.bind(LaplacianMatrixBuilder.class).to(RegularLaplacianMatrixBuilder.class);

        config_reg.bind(VectorSimilarity.class).to(CosineVectorSimilarity.class);
        config_diff_n.bind(VectorSimilarity.class).to(DiffusedCosineVectorSimilarity.class);
        config_diff.bind(VectorSimilarity.class).to(DiffusedCosineVectorSimilarity.class);



        AlgorithmInstance regular_algo = new AlgorithmInstance("regular_" + vectorSimilarityMeasure + "_similarity_itemitemCF",
                config_reg);
        AlgorithmInstance diffusion_norm_algo = new AlgorithmInstance("diffusion_norm_" + vectorSimilarityMeasure + "_similarityitemitemCF",
                config_diff_n);
        AlgorithmInstance diffusion_algo = new AlgorithmInstance("diffusion_" + vectorSimilarityMeasure + "_similarityitemitemCF",
                config_diff);


        //set to run with n threads
        Properties EvalProps = new Properties();
        EvalProps.setProperty(EvalConfig.THREAD_COUNT_PROPERTY, Integer.toString(numThreads));
        SimpleEvaluator simpleEval = new SimpleEvaluator(EvalProps);
        simpleEval.addAlgorithm(regular_algo);
        simpleEval.addAlgorithm(diffusion_norm_algo);
        simpleEval.addAlgorithm(diffusion_algo);

        return simpleEval;
    }

    private SimpleEvaluator testEval2(int numThreads, double alpha, double thresholdFraction){
        /* no normalization (doesn't matter), itemCF, directed usage similarity, cosine vector similarity */

        LenskitConfiguration config_diff_n = new LenskitConfiguration();
        LenskitConfiguration config_reg = new LenskitConfiguration();
        LenskitConfiguration config_diff = new LenskitConfiguration();

        set_config_itemCF(config_diff_n);
        set_config_itemCF(config_diff);
        set_config_itemCF(config_reg);

        //set normalized settings
        config_diff_n.set(Alpha_nL.class).to(alpha);
        config_diff_n.set(ThresholdFraction.class).to(thresholdFraction);
        config_diff_n.bind(DiffusionModel.class).to(ItemCFDiffusionModel.class);
        config_diff_n.bind(UserUserSimilarityMatrixBuilder.class).to(DirectedUserUserSimilarityMatrixBuilder.class);
        config_diff_n.bind(UtilityMatrixNormalizer.class).to(DoNothingUtilityMatrixNormalizer.class);
        config_diff_n.bind(LaplacianMatrixBuilder.class).to(NormalizedLaplacianMatrixBuilder.class);

        //set non-normalized settings
        config_diff.set(Alpha_nL.class).to(alpha);
        config_diff.set(ThresholdFraction.class).to(thresholdFraction);
        config_diff.bind(DiffusionModel.class).to(ItemCFDiffusionModel.class);
        config_diff.bind(UserUserSimilarityMatrixBuilder.class).to(DirectedUserUserSimilarityMatrixBuilder.class);
        config_diff.bind(UtilityMatrixNormalizer.class).to(DoNothingUtilityMatrixNormalizer.class);
        config_diff.bind(LaplacianMatrixBuilder.class).to(RegularLaplacianMatrixBuilder.class);

        config_reg.bind(VectorSimilarity.class).to(CosineVectorSimilarity.class);
        config_diff_n.bind(VectorSimilarity.class).to(DiffusedCosineVectorSimilarity.class);
        config_diff.bind(VectorSimilarity.class).to(DiffusedCosineVectorSimilarity.class);

        AlgorithmInstance regular_algo = new AlgorithmInstance("regular_" + vectorSimilarityMeasure + "_similarity_itemitemCF",
                config_reg);
        AlgorithmInstance diffusion_norm_algo = new AlgorithmInstance("diffusion_norm_" + vectorSimilarityMeasure + "_similarityitemitemCF",
                config_diff_n);
        AlgorithmInstance diffusion_algo = new AlgorithmInstance("diffusion_" + vectorSimilarityMeasure + "_similarityitemitemCF",
                config_diff);


        //set to run with n threads
        Properties EvalProps = new Properties();
        EvalProps.setProperty(EvalConfig.THREAD_COUNT_PROPERTY, Integer.toString(numThreads));
        SimpleEvaluator simpleEval = new SimpleEvaluator(EvalProps);
        simpleEval.addAlgorithm(regular_algo);
        simpleEval.addAlgorithm(diffusion_norm_algo);
        simpleEval.addAlgorithm(diffusion_algo);

        return simpleEval;
    }

    private SimpleEvaluator testEval3(int numThreads, double alpha, double thresholdFraction){
        /* itemUser normlization, userCF, cosine similarity, cosine vector similarity */

        LenskitConfiguration config_diff_n = new LenskitConfiguration();
        LenskitConfiguration config_reg = new LenskitConfiguration();
        LenskitConfiguration config_diff = new LenskitConfiguration();

        set_config_userCF(config_diff_n);
        set_config_userCF(config_diff);
        set_config_userCF(config_reg);

        //set normalized settings
        config_diff_n.set(Alpha_nL.class).to(alpha);
        config_diff_n.set(ThresholdFraction.class).to(thresholdFraction);
        config_diff_n.bind(DiffusionModel.class).to(UserCFDiffusionModel.class);
        config_diff_n.bind(ItemItemSimilarityMatrixBuilder.class).to(CosineItemItemSimilarityMatrixBuilder.class);
        config_diff_n.bind(UtilityMatrixNormalizer.class).to(ItemUserUtilityMatrixNormalizer.class);
        config_diff_n.bind(LaplacianMatrixBuilder.class).to(NormalizedLaplacianMatrixBuilder.class);

        //set non-normalized settings
        config_diff.set(Alpha_nL.class).to(alpha);
        config_diff.set(ThresholdFraction.class).to(thresholdFraction);
        config_diff.bind(DiffusionModel.class).to(UserCFDiffusionModel.class);
        config_diff.bind(ItemItemSimilarityMatrixBuilder.class).to(CosineItemItemSimilarityMatrixBuilder.class);
        config_diff.bind(UtilityMatrixNormalizer.class).to(ItemUserUtilityMatrixNormalizer.class);
        config_diff.bind(LaplacianMatrixBuilder.class).to(RegularLaplacianMatrixBuilder.class);

        //use cosine vector similarity
        config_reg.bind(VectorSimilarity.class).to(CosineVectorSimilarity.class);
        config_diff_n.bind(VectorSimilarity.class).to(DiffusedCosineVectorSimilarity.class);
        config_diff.bind(VectorSimilarity.class).to(DiffusedCosineVectorSimilarity.class);

        AlgorithmInstance regular_algo = new AlgorithmInstance("regular_" + vectorSimilarityMeasure + "_similarity_useruserCF",
                config_reg);
        AlgorithmInstance diffusion_norm_algo = new AlgorithmInstance("diffusion_norm_" + vectorSimilarityMeasure + "_similarity_useruserCF",
                config_diff_n);
        AlgorithmInstance diffusion_algo = new AlgorithmInstance("diffusion_" + vectorSimilarityMeasure + "_similarity_useruserCF",
                config_diff);

        //set to run with n threads
        Properties EvalProps = new Properties();
        EvalProps.setProperty(EvalConfig.THREAD_COUNT_PROPERTY, Integer.toString(numThreads));
        SimpleEvaluator simpleEval = new SimpleEvaluator(EvalProps);
        simpleEval.addAlgorithm(regular_algo);
        simpleEval.addAlgorithm(diffusion_norm_algo);
        simpleEval.addAlgorithm(diffusion_algo);

        return simpleEval;
    }

    private SimpleEvaluator testEval4(int numThreads, double alpha, double thresholdFraction){
        /* no normalization (doesn't matter), userCF, directed usage similarity, cosine vector similarity */


        LenskitConfiguration config_diff_n = new LenskitConfiguration();
        LenskitConfiguration config_reg = new LenskitConfiguration();
        LenskitConfiguration config_diff = new LenskitConfiguration();

        set_config_userCF(config_diff_n);
        set_config_userCF(config_diff);
        set_config_userCF(config_reg);

        //set normalized settings
        config_diff_n.set(Alpha_nL.class).to(alpha);
        config_diff_n.set(ThresholdFraction.class).to(thresholdFraction);
        config_diff_n.bind(DiffusionModel.class).to(UserCFDiffusionModel.class);
        config_diff_n.bind(ItemItemSimilarityMatrixBuilder.class).to(DirectedItemItemSimilarityMatrixBuilder.class);
        config_diff_n.bind(UtilityMatrixNormalizer.class).to(DoNothingUtilityMatrixNormalizer.class);
        config_diff_n.bind(LaplacianMatrixBuilder.class).to(NormalizedLaplacianMatrixBuilder.class);

        //set non-normalized settings
        config_diff.set(Alpha_nL.class).to(alpha);
        config_diff.set(ThresholdFraction.class).to(thresholdFraction);
        config_diff.bind(DiffusionModel.class).to(UserCFDiffusionModel.class);
        config_diff.bind(ItemItemSimilarityMatrixBuilder.class).to(DirectedItemItemSimilarityMatrixBuilder.class);
        config_diff.bind(UtilityMatrixNormalizer.class).to(DoNothingUtilityMatrixNormalizer.class);
        config_diff.bind(LaplacianMatrixBuilder.class).to(RegularLaplacianMatrixBuilder.class);

        //use cosine vector similarity
        config_reg.bind(VectorSimilarity.class).to(CosineVectorSimilarity.class);
        config_diff_n.bind(VectorSimilarity.class).to(DiffusedCosineVectorSimilarity.class);
        config_diff.bind(VectorSimilarity.class).to(DiffusedCosineVectorSimilarity.class);

        AlgorithmInstance regular_algo = new AlgorithmInstance("regular_" + vectorSimilarityMeasure + "_similarity_useruserCF",
                config_reg);
        AlgorithmInstance diffusion_norm_algo = new AlgorithmInstance("diffusion_norm_" + vectorSimilarityMeasure + "_similarity_useruserCF",
                config_diff_n);
        AlgorithmInstance diffusion_algo = new AlgorithmInstance("diffusion_" + vectorSimilarityMeasure + "_similarity_useruserCF",
                config_diff);

        //set to run with n threads
        Properties EvalProps = new Properties();
        EvalProps.setProperty(EvalConfig.THREAD_COUNT_PROPERTY, Integer.toString(numThreads));
        SimpleEvaluator simpleEval = new SimpleEvaluator(EvalProps);
        simpleEval.addAlgorithm(regular_algo);
        simpleEval.addAlgorithm(diffusion_norm_algo);
        simpleEval.addAlgorithm(diffusion_algo);

        return simpleEval;
    }

    private SimpleEvaluator testEval5(int numThreads, double alpha, double thresholdFraction){
        /*no normalization, itemCF, cosine similarity, cosine vector similarity*/

        LenskitConfiguration config_diff_n = new LenskitConfiguration();
        LenskitConfiguration config_reg = new LenskitConfiguration();
        LenskitConfiguration config_diff = new LenskitConfiguration();

        set_config_itemCF(config_diff_n);
        set_config_itemCF(config_diff);
        set_config_itemCF(config_reg);

        //set normalized settings
        config_diff_n.set(Alpha_nL.class).to(alpha);
        config_diff_n.set(ThresholdFraction.class).to(thresholdFraction);
        config_diff_n.bind(DiffusionModel.class).to(ItemCFDiffusionModel.class);
        config_diff_n.bind(UserUserSimilarityMatrixBuilder.class).to(CosineUserUserSimilarityMatrixBuilder.class);
        config_diff_n.bind(UtilityMatrixNormalizer.class).to(DoNothingUtilityMatrixNormalizer.class);

        //set non-normalized settings
        config_diff.set(Alpha_nL.class).to(alpha);
        config_diff.set(ThresholdFraction.class).to(thresholdFraction);
        config_diff.bind(DiffusionModel.class).to(ItemCFDiffusionModel.class);
        config_diff.bind(UserUserSimilarityMatrixBuilder.class).to(CosineUserUserSimilarityMatrixBuilder.class);
        config_diff.bind(UtilityMatrixNormalizer.class).to(DoNothingUtilityMatrixNormalizer.class);
        config_diff.bind(LaplacianMatrixBuilder.class).to(RegularLaplacianMatrixBuilder.class);

        config_reg.bind(VectorSimilarity.class).to(CosineVectorSimilarity.class);
        config_diff_n.bind(VectorSimilarity.class).to(DiffusedCosineVectorSimilarity.class);
        config_diff.bind(VectorSimilarity.class).to(DiffusedCosineVectorSimilarity.class);

        AlgorithmInstance regular_algo = new AlgorithmInstance("regular_" + vectorSimilarityMeasure + "_similarity_itemitemCF",
                config_reg);
        AlgorithmInstance diffusion_norm_algo = new AlgorithmInstance("diffusion_norm_" + vectorSimilarityMeasure + "_similarityitemitemCF",
                config_diff_n);
        AlgorithmInstance diffusion_algo = new AlgorithmInstance("diffusion_" + vectorSimilarityMeasure + "_similarityitemitemCF",
                config_diff);


        //set to run with n threads
        Properties EvalProps = new Properties();
        EvalProps.setProperty(EvalConfig.THREAD_COUNT_PROPERTY, Integer.toString(numThreads));
        SimpleEvaluator simpleEval = new SimpleEvaluator(EvalProps);
        simpleEval.addAlgorithm(regular_algo);
        simpleEval.addAlgorithm(diffusion_norm_algo);
        simpleEval.addAlgorithm(diffusion_algo);

        return simpleEval;
    }

    private SimpleEvaluator testEval6(int numThreads, double alpha, double thresholdFraction){
        /* no normalization, userCF, cosine similarity, cosine vector similarity */


        LenskitConfiguration config_diff_n = new LenskitConfiguration();
        LenskitConfiguration config_reg = new LenskitConfiguration();
        LenskitConfiguration config_diff = new LenskitConfiguration();

        set_config_userCF(config_diff_n);
        set_config_userCF(config_diff);
        set_config_userCF(config_reg);

        //set normalized settings
        config_diff_n.set(Alpha_nL.class).to(alpha);
        config_diff_n.set(ThresholdFraction.class).to(thresholdFraction);
        config_diff_n.bind(DiffusionModel.class).to(UserCFDiffusionModel.class);
        config_diff_n.bind(ItemItemSimilarityMatrixBuilder.class).to(CosineItemItemSimilarityMatrixBuilder.class);
        config_diff_n.bind(UtilityMatrixNormalizer.class).to(DoNothingUtilityMatrixNormalizer.class);
        config_diff_n.bind(LaplacianMatrixBuilder.class).to(NormalizedLaplacianMatrixBuilder.class);

        //set non-normalized settings
        config_diff.set(Alpha_nL.class).to(alpha);
        config_diff.set(ThresholdFraction.class).to(thresholdFraction);
        config_diff.bind(DiffusionModel.class).to(UserCFDiffusionModel.class);
        config_diff.bind(ItemItemSimilarityMatrixBuilder.class).to(CosineItemItemSimilarityMatrixBuilder.class);
        config_diff.bind(UtilityMatrixNormalizer.class).to(DoNothingUtilityMatrixNormalizer.class);
        config_diff.bind(LaplacianMatrixBuilder.class).to(RegularLaplacianMatrixBuilder.class);

        //use cosine vector similarity
        config_reg.bind(VectorSimilarity.class).to(CosineVectorSimilarity.class);
        config_diff_n.bind(VectorSimilarity.class).to(DiffusedCosineVectorSimilarity.class);
        config_diff.bind(VectorSimilarity.class).to(DiffusedCosineVectorSimilarity.class);

        AlgorithmInstance regular_algo = new AlgorithmInstance("regular_" + vectorSimilarityMeasure + "_similarity_useruserCF",
                config_reg);
        AlgorithmInstance diffusion_norm_algo = new AlgorithmInstance("diffusion_norm_" + vectorSimilarityMeasure + "_similarity_useruserCF",
                config_diff_n);
        AlgorithmInstance diffusion_algo = new AlgorithmInstance("diffusion_" + vectorSimilarityMeasure + "_similarity_useruserCF",
                config_diff);

        //set to run with n threads
        Properties EvalProps = new Properties();
        EvalProps.setProperty(EvalConfig.THREAD_COUNT_PROPERTY, Integer.toString(numThreads));
        SimpleEvaluator simpleEval = new SimpleEvaluator(EvalProps);
        simpleEval.addAlgorithm(regular_algo);
        simpleEval.addAlgorithm(diffusion_norm_algo);
        simpleEval.addAlgorithm(diffusion_algo);

        return simpleEval;
    }

    private SimpleEvaluator testEval7(int numThreads, double alpha, double thresholdFraction){
        /*normalize by user, UserCF, cosine similarity, vector cosine similarity*/

        LenskitConfiguration config_diff_n = new LenskitConfiguration();
        LenskitConfiguration config_reg = new LenskitConfiguration();
        LenskitConfiguration config_diff = new LenskitConfiguration();

        set_config_userCF(config_diff_n);
        set_config_userCF(config_diff);
        set_config_userCF(config_reg);

        //set normalized settings
        config_diff_n.set(Alpha_nL.class).to(alpha);
        config_diff_n.set(ThresholdFraction.class).to(thresholdFraction);
        config_diff_n.bind(DiffusionModel.class).to(UserCFDiffusionModel.class);
        config_diff_n.bind(ItemItemSimilarityMatrixBuilder.class).to(CosineItemItemSimilarityMatrixBuilder.class);
        config_diff_n.bind(UtilityMatrixNormalizer.class).to(UserUtilityMatrixNormalizer.class);
        config_diff_n.bind(LaplacianMatrixBuilder.class).to(NormalizedLaplacianMatrixBuilder.class);

        //set non-normalized settings
        config_diff.set(Alpha_nL.class).to(alpha);
        config_diff.set(ThresholdFraction.class).to(thresholdFraction);
        config_diff.bind(DiffusionModel.class).to(UserCFDiffusionModel.class);
        config_diff.bind(ItemItemSimilarityMatrixBuilder.class).to(CosineItemItemSimilarityMatrixBuilder.class);
        config_diff.bind(UtilityMatrixNormalizer.class).to(UserUtilityMatrixNormalizer.class);
        config_diff.bind(LaplacianMatrixBuilder.class).to(RegularLaplacianMatrixBuilder.class);

        //use cosine vector similarity
        config_reg.bind(VectorSimilarity.class).to(CosineVectorSimilarity.class);
        config_diff_n.bind(VectorSimilarity.class).to(DiffusedCosineVectorSimilarity.class);
        config_diff.bind(VectorSimilarity.class).to(DiffusedCosineVectorSimilarity.class);

        AlgorithmInstance regular_algo = new AlgorithmInstance("regular_" + vectorSimilarityMeasure + "_similarity_useruserCF",
                config_reg);
        AlgorithmInstance diffusion_norm_algo = new AlgorithmInstance("diffusion_norm_" + vectorSimilarityMeasure + "_similarity_useruserCF",
                config_diff_n);
        AlgorithmInstance diffusion_algo = new AlgorithmInstance("diffusion_" + vectorSimilarityMeasure + "_similarity_useruserCF",
                config_diff);

        //set to run with n threads
        Properties EvalProps = new Properties();
        EvalProps.setProperty(EvalConfig.THREAD_COUNT_PROPERTY, Integer.toString(numThreads));
        SimpleEvaluator simpleEval = new SimpleEvaluator(EvalProps);
        simpleEval.addAlgorithm(regular_algo);
        simpleEval.addAlgorithm(diffusion_norm_algo);
        simpleEval.addAlgorithm(diffusion_algo);

        return simpleEval;
    }

    private SimpleEvaluator testEval8(int numThreads, double alpha, double thresholdFraction){
        /* User normalization, itemCF, cosine similarity, cosine vector similarity */


        LenskitConfiguration config_diff_n = new LenskitConfiguration();
        LenskitConfiguration config_reg = new LenskitConfiguration();
        LenskitConfiguration config_diff = new LenskitConfiguration();

        set_config_itemCF(config_diff_n);
        set_config_itemCF(config_diff);
        set_config_itemCF(config_reg);

        //set normalized settings
        config_diff_n.set(Alpha_nL.class).to(alpha);
        config_diff_n.set(ThresholdFraction.class).to(thresholdFraction);
        config_diff_n.bind(DiffusionModel.class).to(ItemCFDiffusionModel.class);
        config_diff_n.bind(UserUserSimilarityMatrixBuilder.class).to(CosineUserUserSimilarityMatrixBuilder.class);
        config_diff_n.bind(UtilityMatrixNormalizer.class).to(UserUtilityMatrixNormalizer.class);

        //set non-normalized settings
        config_diff.set(Alpha_nL.class).to(alpha);
        config_diff.set(ThresholdFraction.class).to(thresholdFraction);
        config_diff.bind(DiffusionModel.class).to(ItemCFDiffusionModel.class);
        config_diff.bind(UserUserSimilarityMatrixBuilder.class).to(CosineUserUserSimilarityMatrixBuilder.class);
        config_diff.bind(UtilityMatrixNormalizer.class).to(UserUtilityMatrixNormalizer.class);
        config_diff.bind(LaplacianMatrixBuilder.class).to(RegularLaplacianMatrixBuilder.class);

        config_reg.bind(VectorSimilarity.class).to(CosineVectorSimilarity.class);
        config_diff_n.bind(VectorSimilarity.class).to(DiffusedCosineVectorSimilarity.class);
        config_diff.bind(VectorSimilarity.class).to(DiffusedCosineVectorSimilarity.class);



        AlgorithmInstance regular_algo = new AlgorithmInstance("regular_" + vectorSimilarityMeasure + "_similarity_itemitemCF",
                config_reg);
        AlgorithmInstance diffusion_norm_algo = new AlgorithmInstance("diffusion_norm_" + vectorSimilarityMeasure + "_similarityitemitemCF",
                config_diff_n);
        AlgorithmInstance diffusion_algo = new AlgorithmInstance("diffusion_" + vectorSimilarityMeasure + "_similarityitemitemCF",
                config_diff);


        //set to run with n threads
        Properties EvalProps = new Properties();
        EvalProps.setProperty(EvalConfig.THREAD_COUNT_PROPERTY, Integer.toString(numThreads));
        SimpleEvaluator simpleEval = new SimpleEvaluator(EvalProps);
        simpleEval.addAlgorithm(regular_algo);
        simpleEval.addAlgorithm(diffusion_norm_algo);
        simpleEval.addAlgorithm(diffusion_algo);

        return simpleEval;
    }

    private SimpleEvaluator testEval9(int numThreads, double alpha, double thresholdFraction){
        /* Item normalization, itemCF, cosine similarity, cosine vector similarity */


        LenskitConfiguration config_diff_n = new LenskitConfiguration();
        LenskitConfiguration config_reg = new LenskitConfiguration();
        LenskitConfiguration config_diff = new LenskitConfiguration();

        set_config_itemCF(config_diff_n);
        set_config_itemCF(config_diff);
        set_config_itemCF(config_reg);

        //set normalized settings
        config_diff_n.set(Alpha_nL.class).to(alpha);
        config_diff_n.set(ThresholdFraction.class).to(thresholdFraction);
        config_diff_n.bind(DiffusionModel.class).to(ItemCFDiffusionModel.class);
        config_diff_n.bind(UserUserSimilarityMatrixBuilder.class).to(CosineUserUserSimilarityMatrixBuilder.class);
        config_diff_n.bind(UtilityMatrixNormalizer.class).to(ItemUtilityMatrixNormalizer.class);

        //set non-normalized settings
        config_diff.set(Alpha_nL.class).to(alpha);
        config_diff.set(ThresholdFraction.class).to(thresholdFraction);
        config_diff.bind(DiffusionModel.class).to(ItemCFDiffusionModel.class);
        config_diff.bind(UserUserSimilarityMatrixBuilder.class).to(CosineUserUserSimilarityMatrixBuilder.class);
        config_diff.bind(UtilityMatrixNormalizer.class).to(ItemUtilityMatrixNormalizer.class);
        config_diff.bind(LaplacianMatrixBuilder.class).to(RegularLaplacianMatrixBuilder.class);

        config_reg.bind(VectorSimilarity.class).to(CosineVectorSimilarity.class);
        config_diff_n.bind(VectorSimilarity.class).to(DiffusedCosineVectorSimilarity.class);
        config_diff.bind(VectorSimilarity.class).to(DiffusedCosineVectorSimilarity.class);



        AlgorithmInstance regular_algo = new AlgorithmInstance("regular_" + vectorSimilarityMeasure + "_similarity_itemitemCF",
                config_reg);
        AlgorithmInstance diffusion_norm_algo = new AlgorithmInstance("diffusion_norm_" + vectorSimilarityMeasure + "_similarityitemitemCF",
                config_diff_n);
        AlgorithmInstance diffusion_algo = new AlgorithmInstance("diffusion_" + vectorSimilarityMeasure + "_similarityitemitemCF",
                config_diff);


        //set to run with n threads
        Properties EvalProps = new Properties();
        EvalProps.setProperty(EvalConfig.THREAD_COUNT_PROPERTY, Integer.toString(numThreads));
        SimpleEvaluator simpleEval = new SimpleEvaluator(EvalProps);
        simpleEval.addAlgorithm(regular_algo);
        simpleEval.addAlgorithm(diffusion_norm_algo);
        simpleEval.addAlgorithm(diffusion_algo);

        return simpleEval;
    }

    private SimpleEvaluator testEval10(int numThreads, double alpha, double thresholdFraction){
        /*normalize by item, UserCF, cosine similarity, vector cosine similarity*/

        LenskitConfiguration config_diff_n = new LenskitConfiguration();
        LenskitConfiguration config_reg = new LenskitConfiguration();
        LenskitConfiguration config_diff = new LenskitConfiguration();

        set_config_userCF(config_diff_n);
        set_config_userCF(config_diff);
        set_config_userCF(config_reg);

        //set normalized settings
        config_diff_n.set(Alpha_nL.class).to(alpha);
        config_diff_n.set(ThresholdFraction.class).to(thresholdFraction);
        config_diff_n.bind(DiffusionModel.class).to(UserCFDiffusionModel.class);
        config_diff_n.bind(ItemItemSimilarityMatrixBuilder.class).to(CosineItemItemSimilarityMatrixBuilder.class);
        config_diff_n.bind(UtilityMatrixNormalizer.class).to(ItemUtilityMatrixNormalizer.class);
        config_diff_n.bind(LaplacianMatrixBuilder.class).to(NormalizedLaplacianMatrixBuilder.class);

        //set non-normalized settings
        config_diff.set(Alpha_nL.class).to(alpha);
        config_diff.set(ThresholdFraction.class).to(thresholdFraction);
        config_diff.bind(DiffusionModel.class).to(UserCFDiffusionModel.class);
        config_diff.bind(ItemItemSimilarityMatrixBuilder.class).to(CosineItemItemSimilarityMatrixBuilder.class);
        config_diff.bind(UtilityMatrixNormalizer.class).to(ItemUtilityMatrixNormalizer.class);
        config_diff.bind(LaplacianMatrixBuilder.class).to(RegularLaplacianMatrixBuilder.class);

        //use cosine vector similarity
        config_reg.bind(VectorSimilarity.class).to(CosineVectorSimilarity.class);
        config_diff_n.bind(VectorSimilarity.class).to(DiffusedCosineVectorSimilarity.class);
        config_diff.bind(VectorSimilarity.class).to(DiffusedCosineVectorSimilarity.class);

        AlgorithmInstance regular_algo = new AlgorithmInstance("regular_" + vectorSimilarityMeasure + "_similarity_useruserCF",
                config_reg);
        AlgorithmInstance diffusion_norm_algo = new AlgorithmInstance("diffusion_norm_" + vectorSimilarityMeasure + "_similarity_useruserCF",
                config_diff_n);
        AlgorithmInstance diffusion_algo = new AlgorithmInstance("diffusion_" + vectorSimilarityMeasure + "_similarity_useruserCF",
                config_diff);

        //set to run with n threads
        Properties EvalProps = new Properties();
        EvalProps.setProperty(EvalConfig.THREAD_COUNT_PROPERTY, Integer.toString(numThreads));
        SimpleEvaluator simpleEval = new SimpleEvaluator(EvalProps);
        simpleEval.addAlgorithm(regular_algo);
        simpleEval.addAlgorithm(diffusion_norm_algo);
        simpleEval.addAlgorithm(diffusion_algo);

        return simpleEval;
    }

    private SimpleEvaluator testEval11(int numThreads, double alpha, double thresholdFraction){
        /* itemCF, pearsons correlation, cosine vector similarity */

        LenskitConfiguration config_diff_n = new LenskitConfiguration();
        LenskitConfiguration config_reg = new LenskitConfiguration();
        LenskitConfiguration config_diff = new LenskitConfiguration();

        set_config_itemCF(config_diff_n);
        set_config_itemCF(config_diff);
        set_config_itemCF(config_reg);

        //set normalized settings
        config_diff_n.set(Alpha_nL.class).to(alpha);
        config_diff_n.set(ThresholdFraction.class).to(thresholdFraction);
        config_diff_n.bind(DiffusionModel.class).to(ItemCFDiffusionModel.class);
        config_diff_n.bind(UserUserSimilarityMatrixBuilder.class).to(PearsonCorrelationUserUserSimilarityMatrixBuilder.class);
        config_diff_n.bind(UtilityMatrixNormalizer.class).to(DoNothingUtilityMatrixNormalizer.class);

        //set non-normalized settings
        config_diff.set(Alpha_nL.class).to(alpha);
        config_diff.set(ThresholdFraction.class).to(thresholdFraction);
        config_diff.bind(DiffusionModel.class).to(ItemCFDiffusionModel.class);
        config_diff.bind(UserUserSimilarityMatrixBuilder.class).to(PearsonCorrelationUserUserSimilarityMatrixBuilder.class);
        config_diff.bind(UtilityMatrixNormalizer.class).to(DoNothingUtilityMatrixNormalizer.class);
        config_diff.bind(LaplacianMatrixBuilder.class).to(RegularLaplacianMatrixBuilder.class);

        config_reg.bind(VectorSimilarity.class).to(CosineVectorSimilarity.class);
        config_diff_n.bind(VectorSimilarity.class).to(DiffusedCosineVectorSimilarity.class);
        config_diff.bind(VectorSimilarity.class).to(DiffusedCosineVectorSimilarity.class);



        AlgorithmInstance regular_algo = new AlgorithmInstance("regular_" + vectorSimilarityMeasure + "_similarity_itemitemCF",
                config_reg);
        AlgorithmInstance diffusion_norm_algo = new AlgorithmInstance("diffusion_norm_" + vectorSimilarityMeasure + "_similarityitemitemCF",
                config_diff_n);
        AlgorithmInstance diffusion_algo = new AlgorithmInstance("diffusion_" + vectorSimilarityMeasure + "_similarityitemitemCF",
                config_diff);


        //set to run with n threads
        Properties EvalProps = new Properties();
        EvalProps.setProperty(EvalConfig.THREAD_COUNT_PROPERTY, Integer.toString(numThreads));
        SimpleEvaluator simpleEval = new SimpleEvaluator(EvalProps);
        simpleEval.addAlgorithm(regular_algo);
        simpleEval.addAlgorithm(diffusion_norm_algo);
        simpleEval.addAlgorithm(diffusion_algo);

        return simpleEval;
    }

    private SimpleEvaluator testEval12(int numThreads, double alpha, double thresholdFraction){
        /*no normalization, UserCF, pearson correlation, vector cosine similarity*/
        System.out.println("Pearson Correlation UserCF cosine vectorsim");
        LenskitConfiguration config_diff_n = new LenskitConfiguration();
        LenskitConfiguration config_reg = new LenskitConfiguration();
        LenskitConfiguration config_diff = new LenskitConfiguration();

        set_config_userCF(config_diff_n);
        set_config_userCF(config_diff);
        set_config_userCF(config_reg);

        //set normalized settings
        config_diff_n.set(Alpha_nL.class).to(alpha);
        config_diff_n.set(ThresholdFraction.class).to(thresholdFraction);
        config_diff_n.bind(DiffusionModel.class).to(UserCFDiffusionModel.class);
        config_diff_n.bind(ItemItemSimilarityMatrixBuilder.class).to(PearsonCorrelationItemItemSimilarityMatrixBuilder.class);
        config_diff_n.bind(UtilityMatrixNormalizer.class).to(DoNothingUtilityMatrixNormalizer.class);
        config_diff_n.bind(LaplacianMatrixBuilder.class).to(NormalizedLaplacianMatrixBuilder.class);

        //set non-normalized settings
        config_diff.set(Alpha_nL.class).to(alpha);
        config_diff.set(ThresholdFraction.class).to(thresholdFraction);
        config_diff.bind(DiffusionModel.class).to(UserCFDiffusionModel.class);
        config_diff.bind(ItemItemSimilarityMatrixBuilder.class).to(PearsonCorrelationItemItemSimilarityMatrixBuilder.class);
        config_diff.bind(UtilityMatrixNormalizer.class).to(DoNothingUtilityMatrixNormalizer.class);
        config_diff.bind(LaplacianMatrixBuilder.class).to(RegularLaplacianMatrixBuilder.class);

        //use cosine vector similarity
        config_reg.bind(VectorSimilarity.class).to(CosineVectorSimilarity.class);
        config_diff_n.bind(VectorSimilarity.class).to(DiffusedCosineVectorSimilarity.class);
        config_diff.bind(VectorSimilarity.class).to(DiffusedCosineVectorSimilarity.class);

        AlgorithmInstance regular_algo = new AlgorithmInstance("regular_" + vectorSimilarityMeasure + "_similarity_useruserCF",
                config_reg);
        AlgorithmInstance diffusion_norm_algo = new AlgorithmInstance("diffusion_norm_" + vectorSimilarityMeasure + "_similarity_useruserCF",
                config_diff_n);
        AlgorithmInstance diffusion_algo = new AlgorithmInstance("diffusion_" + vectorSimilarityMeasure + "_similarity_useruserCF",
                config_diff);

        //set to run with n threads
        Properties EvalProps = new Properties();
        EvalProps.setProperty(EvalConfig.THREAD_COUNT_PROPERTY, Integer.toString(numThreads));
        SimpleEvaluator simpleEval = new SimpleEvaluator(EvalProps);
        simpleEval.addAlgorithm(regular_algo);
        simpleEval.addAlgorithm(diffusion_norm_algo);
        simpleEval.addAlgorithm(diffusion_algo);

        return simpleEval;
    }



    public MainTester(String[] args) {
        System.out.println("hello");
        //runslopeOne();
    }

    public void runsingle(){
        SimpleEvaluator simpleEval;
        System.out.println("SVD!!!");

        /*
        config.set(FeatureCount.class).to(45);
        config.set(RegularizationTerm.class).to(0.0005);
         */
        int featureCounts [] = {55,60,65,70};
        double regularizations [] = {0.0001,0.0005,0.001};

        for (int featureCount:featureCounts){
            for (double reg:regularizations){
                //create evaluator object
                simpleEval = SVDEval(4,featureCount,reg);

                //construct data source
                File in = new File(dataFileName);
                CSVDataSourceBuilder builder = new CSVDataSourceBuilder(in);
                builder.setDelimiter("\t");
                CSVDataSource dat = builder.build();

                //use 5-fold CV
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

                File out = new File("SVD_results" + "reg" + String.valueOf(featureCount) + "_featcount"+String.valueOf(reg)+ ".csv");
                simpleEval.setOutput(out);

                try{
                    simpleEval.call();
                } catch (Exception e){
                    System.out.println(e.getMessage());
                }
            }
        }


    }

    public void runslopeOne(){

        double dampings [] = {1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5};
        for (double damping:dampings){
            LenskitConfiguration slope_config = new LenskitConfiguration();
            slope_config.bind(ItemScorer.class).to(SlopeOneItemScorer.class);
            slope_config.bind(SlopeOneItemScorer.class).to(WeightedSlopeOneItemScorer.class);
            slope_config.bind(BaselineScorer.class, ItemScorer.class).to(UserMeanItemScorer.class);
            slope_config.bind(UserMeanBaseline.class, ItemScorer.class).to(ItemMeanRatingItemScorer.class);
            slope_config.set(DeviationDamping.class).to(damping);

            AlgorithmInstance slopeone = new AlgorithmInstance("WeightedSlopeOne"+String.valueOf(damping), slope_config);
            Properties EvalProps = new Properties();
            EvalProps.setProperty(EvalConfig.THREAD_COUNT_PROPERTY, Integer.toString(4));
            SimpleEvaluator simpleEval = new SimpleEvaluator(EvalProps);
            simpleEval.addAlgorithm(slopeone);

            //construct data source
            File in = new File(dataFileName);
            CSVDataSourceBuilder builder = new CSVDataSourceBuilder(in);
            builder.setDelimiter("\t");
            CSVDataSource dat = builder.build();

            //use 5-fold CV
            simpleEval.addDataset(dat,5);

            //Add metrics
            RMSEPredictMetric rmse = new RMSEPredictMetric();
            CoveragePredictMetric cover = new CoveragePredictMetric();
            NDCGPredictMetric ndcg = new NDCGPredictMetric();
            MAEPredictMetric mae = new MAEPredictMetric();

            /*
            slope_config.bind(PreferenceDomain.class).to(new PreferenceDomainBuilder(1, 5)
                    .setPrecision(1)
                    .build());
            */
            simpleEval.addMetric(rmse);
            simpleEval.addMetric(cover);
            simpleEval.addMetric(ndcg);
            simpleEval.addMetric(mae);

            File out = new File("WeightedSlopeOne_results" + "deviationdamping_" + String.valueOf(damping) + ".csv");
            simpleEval.setOutput(out);

            try{
                simpleEval.call();
            } catch (Exception e){
                System.out.println(e.getMessage());
            }

        }

    }

    public void run() {
        System.out.println("Hi");
        double alphas [] = {0.5};
        double threshold_fractions [] = {0.1,1.0};

        for (double alpha:alphas){
            for (double threshold_frac:threshold_fractions){
                SimpleEvaluator simpleEval;

                //create evaluator object
                simpleEval = testEval4(8, alpha, threshold_frac);

                //construct data source
                File in = new File(dataFileName);
                CSVDataSourceBuilder builder = new CSVDataSourceBuilder(in);
                builder.setDelimiter("\t");
                CSVDataSource dat = builder.build();

                //use 5-fold CV
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

                File out = new File("cosinenonnormalized_" + "vectorsim_" + vectorSimilarityMeasure + "_threshold_" + Double.toString(threshold_frac) +
                                    "_alpha_" + Double.toString(alpha)+".csv");
                simpleEval.setOutput(out);

                try{
                    simpleEval.call();
                } catch (Exception e){
                    System.out.println(e.getMessage());
                }
            }
        }
    }
}

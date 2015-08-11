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
package org.grouplens.lenskit.hello;

import org.grouplens.lenskit.ItemScorer;
import org.grouplens.lenskit.baseline.BaselineScorer;
import org.grouplens.lenskit.baseline.ItemMeanRatingItemScorer;
import org.grouplens.lenskit.baseline.UserMeanBaseline;
import org.grouplens.lenskit.baseline.UserMeanItemScorer;
import org.grouplens.lenskit.core.LenskitConfiguration;
import org.grouplens.lenskit.eval.EvalConfig;
import org.grouplens.lenskit.eval.algorithm.AlgorithmInstance;
import org.grouplens.lenskit.eval.data.CSVDataSource;
import org.grouplens.lenskit.eval.data.CSVDataSourceBuilder;
import org.grouplens.lenskit.eval.metrics.predict.CoveragePredictMetric;
import org.grouplens.lenskit.eval.metrics.predict.MAEPredictMetric;
import org.grouplens.lenskit.eval.metrics.predict.NDCGPredictMetric;
import org.grouplens.lenskit.eval.metrics.predict.RMSEPredictMetric;
import org.grouplens.lenskit.eval.traintest.SimpleEvaluator;
import org.grouplens.lenskit.iterative.IterationCount;
import org.grouplens.lenskit.iterative.RegularizationTerm;
import org.grouplens.lenskit.knn.NeighborhoodSize;
import org.grouplens.lenskit.knn.item.ItemItemScorer;
import org.grouplens.lenskit.knn.item.ItemSimilarity;
import org.grouplens.lenskit.knn.user.NeighborFinder;
import org.grouplens.lenskit.knn.user.SnapshotNeighborFinder;
import org.grouplens.lenskit.knn.user.UserSimilarity;
import org.grouplens.lenskit.knn.user.UserUserItemScorer;
import org.grouplens.lenskit.mf.funksvd.FunkSVDItemScorer;
import org.grouplens.lenskit.transform.normalize.BaselineSubtractingUserVectorNormalizer;
import org.grouplens.lenskit.transform.normalize.UserVectorNormalizer;
import org.grouplens.lenskit.vectors.similarity.*;

import java.io.File;
import java.util.Properties;

/**
 * Demonstration app for LensKit. This application builds an item-item CF model
 * from a CSV file, then generates recommendations for a user.
 *
 * Usage: java org.grouplens.lenskit.hello.HelloLenskit ratings.csv user
 */
public class HelloLenskit implements Runnable {
    public static void main(String[] args) {
        HelloLenskit hello = new HelloLenskit(args);
        try {
            hello.run();
        } catch (RuntimeException e) {
            System.err.println(e.toString());
            e.printStackTrace(System.err);
            System.exit(1);
        }
    }

    private String dataFileName = "ml-100k/u.data";
    private String resultsFileName = "results.csv";
    private String vectorSimilarityMeasure = "pearson";
    private String method = "itemitemCF";
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
    }

    private void set_config_FunkSVD(LenskitConfiguration config){
        config.bind(ItemScorer.class).to(FunkSVDItemScorer.class);
        config.set(IterationCount.class).to(100);
        config.set(RegularizationTerm.class).to(0.002);
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

    private SimpleEvaluator testEval(int numThreads){
        LenskitConfiguration config_diff_n = new LenskitConfiguration();
        LenskitConfiguration config_reg = new LenskitConfiguration();

        set_config_itemCF(config_diff_n);
        config_diff_n.set(Alpha.class).to(4.0);

        if (vectorSimilarityMeasure.equalsIgnoreCase("cosine")){
            config_reg.bind(VectorSimilarity.class).to(CosineVectorSimilarity.class);
            config_diff_n.bind(VectorSimilarity.class).to(DiffusedCosineVectorSimilarity.class);
        } else {
            config_reg.bind(VectorSimilarity.class).to(PearsonCorrelation.class);
            config_diff_n.bind(VectorSimilarity.class).to(DiffusedPearsonCorrelation.class);
        }

        set_config_itemCF(config_reg);

        AlgorithmInstance regular_algo = new AlgorithmInstance("regular_" + vectorSimilarityMeasure + "_similarity_itemitemCF", config_reg);
        AlgorithmInstance diffusion_norm_algo = new AlgorithmInstance("diffusion_norm_" + vectorSimilarityMeasure + "_similarityitemitemCF", config_diff_n);

        //set to run with n threads
        Properties EvalProps = new Properties();
        EvalProps.setProperty(EvalConfig.THREAD_COUNT_PROPERTY, Integer.toString(numThreads));
        SimpleEvaluator simpleEval = new SimpleEvaluator(EvalProps);
        simpleEval.addAlgorithm(regular_algo);
        simpleEval.addAlgorithm(diffusion_norm_algo);

        return simpleEval;
    }


    public HelloLenskit(String[] args) {
        if (args.length >= 5) {
            System.out.println("Running test with custom settings");
            dataFileName = args[0];
            resultsFileName = args[1];
            vectorSimilarityMeasure = args[2];
            //set num neighbours
            numNeighbours = Integer.parseInt(args[3]);
            method = args[4];
            System.out.println("data: " + dataFileName + ", results: " + resultsFileName +
                    " vector_similarity: " + vectorSimilarityMeasure + "num_neighbours: " + Integer.toString(numNeighbours) +
                    "method: " + method + "\n");

        } else {
            System.out.println("Running test with default settings");
            System.out.println("data: " + dataFileName + ", results: " + resultsFileName +
                    " vector_similarity: " + vectorSimilarityMeasure + "\n");
        }

    }


    public void run() {

        SimpleEvaluator simpleEval;

        //add some double diffusion
        simpleEval = testEval(4);

        /*
        //add some funk SVD
        LenskitConfiguration config_SVD = new LenskitConfiguration();
        set_config_FunkSVD(config_SVD);
        AlgorithmInstance SVD_algo = new AlgorithmInstance("FunkSVD", config_SVD);
        simpleEval.addAlgorithm(SVD_algo);
        */

        File in = new File(dataFileName);
        CSVDataSourceBuilder builder = new CSVDataSourceBuilder(in);
        builder.setDelimiter("\t");
        CSVDataSource dat = builder.build();

        simpleEval.addDataset(dat,5);
        RMSEPredictMetric rmse = new RMSEPredictMetric();
        CoveragePredictMetric cover = new CoveragePredictMetric();
        NDCGPredictMetric ndcg = new NDCGPredictMetric();
        MAEPredictMetric mae = new MAEPredictMetric();

        simpleEval.addMetric(rmse);
        simpleEval.addMetric(cover);
        simpleEval.addMetric(ndcg);
        simpleEval.addMetric(mae);

        File out = new File(resultsFileName);
        simpleEval.setOutput(out);

        try{
            simpleEval.call();
        } catch (Exception e){
            System.out.println(e.getMessage());
        }

    }
}

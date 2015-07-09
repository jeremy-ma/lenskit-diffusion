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

import org.codehaus.groovy.tools.shell.Command;
import org.grouplens.lenskit.ItemRecommender;
import org.grouplens.lenskit.ItemScorer;
import org.grouplens.lenskit.Recommender;
import org.grouplens.lenskit.RecommenderBuildException;
import org.grouplens.lenskit.baseline.BaselineScorer;
import org.grouplens.lenskit.baseline.ItemMeanRatingItemScorer;
import org.grouplens.lenskit.baseline.UserMeanBaseline;
import org.grouplens.lenskit.baseline.UserMeanItemScorer;
import org.grouplens.lenskit.core.LenskitConfiguration;
import org.grouplens.lenskit.core.LenskitRecommender;
import org.grouplens.lenskit.data.dao.EventDAO;
import org.grouplens.lenskit.data.dao.SimpleFileRatingDAO;
import org.grouplens.lenskit.eval.algorithm.AlgorithmInstance;
import org.grouplens.lenskit.eval.data.CSVDataSource;
import org.grouplens.lenskit.eval.data.CSVDataSourceBuilder;
import org.grouplens.lenskit.eval.data.crossfold.CrossfoldTask;
import org.grouplens.lenskit.eval.data.traintest.GenericTTDataBuilder;
import org.grouplens.lenskit.eval.data.traintest.GenericTTDataSet;
import org.grouplens.lenskit.eval.data.traintest.TTDataSet;
import org.grouplens.lenskit.eval.metrics.predict.*;
import org.grouplens.lenskit.eval.script.EvalScript;
import org.grouplens.lenskit.eval.traintest.MetricFactory;
import org.grouplens.lenskit.eval.traintest.SimpleEvaluator;
import org.grouplens.lenskit.eval.traintest.TrainTestEvalTask;
import org.grouplens.lenskit.knn.NeighborhoodSize;
import org.grouplens.lenskit.knn.user.NeighborFinder;
import org.grouplens.lenskit.knn.user.SnapshotNeighborFinder;
import org.grouplens.lenskit.knn.user.UserUserItemScorer;
import org.grouplens.lenskit.scored.ScoredId;
import org.grouplens.lenskit.transform.normalize.BaselineSubtractingUserVectorNormalizer;
import org.grouplens.lenskit.transform.normalize.MeanCenteringVectorNormalizer;
import org.grouplens.lenskit.transform.normalize.UserVectorNormalizer;
import org.grouplens.lenskit.transform.normalize.VectorNormalizer;
import org.grouplens.lenskit.vectors.similarity.*;
import org.grouplens.lenskit.vectors.similarity.DiffusedPearsonCorrelation;

import javax.activation.DataSource;
import java.io.File;
import java.util.ArrayList;
import java.util.List;

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
    private String vectorSimilarityMeasure = "cosine";

    private static void set_config(LenskitConfiguration config){
        config.bind(ItemScorer.class)
                .to(UserUserItemScorer.class);
        config.bind(BaselineScorer.class, ItemScorer.class)
                .to(UserMeanItemScorer.class);
        config.bind(UserMeanBaseline.class, ItemScorer.class)
                .to(ItemMeanRatingItemScorer.class);
        config.bind(UserVectorNormalizer.class)
                .to(BaselineSubtractingUserVectorNormalizer.class);
        config.set(NeighborhoodSize.class).to(30);
        config.bind(NeighborFinder.class).to(SnapshotNeighborFinder.class);

    }

    private static void set_config_mean_predictor(LenskitConfiguration config){
        config.bind(ItemScorer.class)
                .to(UserMeanItemScorer.class);
        config.bind(BaselineScorer.class, ItemScorer.class)
                .to(UserMeanItemScorer.class);
        config.bind(UserMeanBaseline.class, ItemScorer.class)
                .to(ItemMeanRatingItemScorer.class);
        config.bind(UserVectorNormalizer.class)
                .to(BaselineSubtractingUserVectorNormalizer.class);
        config.set(NeighborhoodSize.class).to(40);
        config.bind(NeighborFinder.class).to(SnapshotNeighborFinder.class);
    }


    public HelloLenskit(String[] args) {
        if (args.length == 3) {
            System.out.println("Running test with custom settings");
            dataFileName = args[0];
            resultsFileName = args[1];
            vectorSimilarityMeasure = args[2];

            System.out.println("data: " + dataFileName + ", results: " + resultsFileName +
                    " vector_similarity: " + vectorSimilarityMeasure + "\n");
        } else {
            System.out.println("Running test with default settings");
            System.out.println("data: " + dataFileName + ", results: " + resultsFileName +
                    " vector_similarity: " + vectorSimilarityMeasure + "\n");
        }


    }

    public void run() {

        LenskitConfiguration config_reg = new LenskitConfiguration();
        set_config(config_reg);
        LenskitConfiguration config_diff = new LenskitConfiguration();
        set_config(config_diff);
        LenskitConfiguration config_diff_n = new LenskitConfiguration();
        set_config(config_diff_n);

        if (vectorSimilarityMeasure.equalsIgnoreCase("cosine")){
            config_reg.bind(VectorSimilarity.class).to(CosineVectorSimilarity.class);
            config_diff.bind(VectorSimilarity.class).to(DiffusionCosineVectorSimilarity.class);
            config_diff_n.bind(VectorSimilarity.class).to(NormDiffusionCosineVectorSimilarity.class);
            config_diff.set(DiffusionMatrixType.class).to("diffusion");
        } else {
            config_reg.bind(VectorSimilarity.class).to(DistanceVectorSimilarity.class);
            config_diff.bind(VectorSimilarity.class).to(DiffusionDistanceVectorSimilarity.class);
            config_diff_n.bind(VectorSimilarity.class).to(NormDiffusionDistanceVectorSimilarity.class);
        }

        AlgorithmInstance regular_algo = new AlgorithmInstance("regular_" + vectorSimilarityMeasure + "_similarity", config_reg);
        AlgorithmInstance diffusion_algo = new AlgorithmInstance("diffusion_" + vectorSimilarityMeasure + "_similarity", config_diff);
        AlgorithmInstance diffusion_norm_algo = new AlgorithmInstance("diffusion_norm_" + vectorSimilarityMeasure + "_similarity", config_diff_n);

        SimpleEvaluator simpleEval = new SimpleEvaluator();
        simpleEval.addAlgorithm(diffusion_algo);
        simpleEval.addAlgorithm(regular_algo);
        simpleEval.addAlgorithm(diffusion_norm_algo);

        File in = new File(dataFileName);
        CSVDataSourceBuilder builder = new CSVDataSourceBuilder(in);
        builder.setDelimiter("\t");
        CSVDataSource dat = builder.build();

        simpleEval.addDataset(dat,5,0.2);
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

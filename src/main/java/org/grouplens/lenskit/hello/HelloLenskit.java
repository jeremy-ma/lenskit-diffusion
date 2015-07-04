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

    private String delimiter = "\t";
    private File inputFile = new File("ratings.dat");
    private List<Long> users;


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
        config.set(NeighborhoodSize.class).to(60);
        config.bind(NeighborFinder.class).to(SnapshotNeighborFinder.class);
    }


    public HelloLenskit(String[] args) {
        int nextArg = 0;
        boolean done = false;
        while (!done && nextArg < args.length) {
            String arg = args[nextArg];
            if (arg.equals("-d")) {
                delimiter = args[nextArg + 1];
                nextArg += 2;
            } else if (arg.startsWith("-")) {
                throw new RuntimeException("unknown option: " + arg);
            } else {
                inputFile = new File(arg);
                nextArg += 1;
                done = true;
            }
        }
        users = new ArrayList<Long>(args.length - nextArg);
        for (; nextArg < args.length; nextArg++) {
            users.add(Long.parseLong(args[nextArg]));
        }
    }

    public void run() {
        // We first need to configure the data access.
        // We will use a simple delimited file; you can use something else like
        // a database (see JDBCRatingDAO).
        /*



        EventDAO dao = new SimpleFileRatingDAO(inputFile, delimiter);
        LenskitConfiguration config = new LenskitConfiguration();
        // Second step is to create the LensKit configuration...
        //LenskitConfiguration config = new LenskitConfiguration();
        // ... configure the data source
        config.addComponent(dao);


        // ... and configure the item scorer.  The bind and set methods
        // are what you use to do that. Here, we want an item-item scorer.
        config.bind(ItemScorer.class)
              .to(UserItemBiasItemScorer.class);

        // let's use personalized mean rating as the baseline/fallback predictor.
        // 2-step process:
        // First, use the user mean rating as the baseline scorer
        config.bind(BaselineScorer.class, ItemScorer.class)
               .to(UserMeanItemScorer.class);
        // Second, use the item mean rating as the base for user means
        config.bind(UserMeanBaseline.class, ItemScorer.class)
              .to(ItemMeanRatingItemScorer.class);
        // and normalize ratings by baseline prior to computing similarities
        config.bind(UserVectorNormalizer.class)
              .to(BaselineSubtractingUserVectorNormalizer.class);


        // ... and configure the item scorer.  The bind and set methods
        // are what you use to do that. Here, we want an item-item scorer.
        config.bind(ItemScorer.class)
                .to(UserUserItemScorer.class);

        // let's use personalized mean rating as the baseline/fallback predictor.
        // 2-step process:
        // First, use the user mean rating as the baseline scorer
        config.bind(BaselineScorer.class, ItemScorer.class)
                .to(UserMeanItemScorer.class);
        // Second, use the item mean rating as the base for user means
         config.bind(UserMeanBaseline.class, ItemScorer.class)
                .to(ItemMeanRatingItemScorer.class);

        // normalize by subtracting the user's mean rating. Added by jeremy

        // and normalize ratings by baseline prior to computing similarities
        config.bind(UserVectorNormalizer.class)
                .to(BaselineSubtractingUserVectorNormalizer.class);


        //set neighbourhood size
        config.set(NeighborhoodSize.class).to(30);

        //THIS IS WHERE THE ALGORITHM IS ACTIVATED
        //change the vector similarity thing
        //config.bind(VectorSimilarity.class).to(DiffusionDistanceVectorSimilarity.class);
        //config.bind(VectorSimilarity.class).to(DistanceVectorSimilarity.class);
        //config.bind(VectorSimilarity.class).to(DodgyDistance.class);
        config.bind(VectorSimilarity.class).to(DiffusionCosineVectorSimilarity.class);
        //config.bind(VectorSimilarity.class).to(CosineVectorSimilarity.class);



        config.bind(NeighborFinder.class).to(SnapshotNeighborFinder.class);

        // There are more parameters, roles, and components that can be set. See the
        // JavaDoc for each recommender algorithm for more information.

        // Now that we have a factory, build a recommender from the configuration
        // and data source. This will compute the similarity matrix and return a recommender
        // that uses it.

        */

        LenskitConfiguration config_reg = new LenskitConfiguration();
        set_config(config_reg);
        config_reg.bind(VectorSimilarity.class).to(CosineVectorSimilarity.class);

        LenskitConfiguration config_diff = new LenskitConfiguration();
        set_config(config_diff);
        config_diff.bind(VectorSimilarity.class).to(DiffusionCosineVectorSimilarity.class);

        LenskitConfiguration config_diff_n = new LenskitConfiguration();
        set_config(config_diff_n);
        config_diff_n.bind(VectorSimilarity.class).to(NormDiffusionCosineVectorSimilarity.class);



        LenskitConfiguration config_mean = new LenskitConfiguration();
        set_config_mean_predictor(config_mean);

        AlgorithmInstance regular_algo = new AlgorithmInstance("regular_cosine", config_reg);
        AlgorithmInstance diffusion_algo = new AlgorithmInstance("diffused_cosine", config_diff);
        AlgorithmInstance diffusion_norm_algo = new AlgorithmInstance("diffusion_norm_cosine", config_diff_n);
        AlgorithmInstance simple_mean_algo = new AlgorithmInstance("simple_mean", config_mean);

        SimpleEvaluator simpleEval = new SimpleEvaluator();
        simpleEval.addAlgorithm(diffusion_algo);
        simpleEval.addAlgorithm(regular_algo);
        simpleEval.addAlgorithm(diffusion_norm_algo);
        //simpleEval.addAlgorithm(simple_mean_algo);

        File in = new File("ml-100k/u.data");
        CSVDataSourceBuilder builder = new CSVDataSourceBuilder(in);
        builder.setDelimiter("\t");
        CSVDataSource dat = builder.build();

        simpleEval.addDataset(dat,5,0.2);
        RMSEPredictMetric rmse = new RMSEPredictMetric();
        CoveragePredictMetric cover = new CoveragePredictMetric();
        NDCGPredictMetric ndcg = new NDCGPredictMetric();

        simpleEval.addMetric(rmse);
        simpleEval.addMetric(cover);
        simpleEval.addMetric(ndcg);

        File out = new File("results.csv");
        simpleEval.setOutput(out);
        try{
            simpleEval.call();
        } catch (Exception e){
            System.out.println(e.getMessage());
        }

    }
}

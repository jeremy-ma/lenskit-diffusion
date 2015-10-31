package org.grouplens.lenskit.diffusion.general;

import org.grouplens.lenskit.ItemScorer;
import org.grouplens.lenskit.baseline.BaselineScorer;
import org.grouplens.lenskit.baseline.MeanDamping;
import org.grouplens.lenskit.baseline.UserMeanBaseline;
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
import org.grouplens.lenskit.knn.NeighborhoodSize;
import org.grouplens.lenskit.knn.user.NeighborFinder;
import org.grouplens.lenskit.transform.normalize.UserVectorNormalizer;
import org.grouplens.lenskit.vectors.similarity.VectorSimilarity;

import java.io.File;
import java.util.Properties;

/**
 * Created by jeremyma on 27/09/15.
 */
public class ParameterSearch {

    private Class itemscorer;
    private Class baselinescorer;
    private Class usermeanbaseline;
    private Class uservectornormalizer;
    private Class neighborhoodfinder;
    private Class diffusionmodel;
    private Class similaritymatrixbuilder;
    private Class utilitymatrixnormalizer;
    private Class laplacianbuilder;
    private Class vectorsimilarity;
    private Class regular_vectorsimilarity;

    private static String dataFileName = "ml-100k/u.data";
    private static double meanDamping = 5.0;

    public ParameterSearch(Class itemscorer, Class baselinescorer, Class usermeanbaseline,
                           Class uservectornormalizer, Class neighborhoodfinder,
                           Class diffusionmodel, Class similaritymatrixbuilder,
                           Class utilitymatrixnormalizer, Class laplacianbuilder, Class vectorsimilarity,
                           Class regular_vectorsimilarity){
        this.itemscorer = itemscorer;
        this.baselinescorer = baselinescorer;
        this.usermeanbaseline = usermeanbaseline;
        this.uservectornormalizer = uservectornormalizer;
        this.neighborhoodfinder = neighborhoodfinder;
        this.diffusionmodel = diffusionmodel;
        this.similaritymatrixbuilder = similaritymatrixbuilder;
        this.utilitymatrixnormalizer = utilitymatrixnormalizer;
        this.laplacianbuilder = laplacianbuilder;
        this.vectorsimilarity = vectorsimilarity;
        this.regular_vectorsimilarity = regular_vectorsimilarity;
    }

    public void runWithParameters(int neighbourhoodsizes [],
                                  double alphas [] , double thresholdFractions [],Double holdoutFraction){
        for (int neighbourhoodsize: neighbourhoodsizes){
            for (double alpha : alphas){
                for (double threshold: thresholdFractions){
                    System.out.println("N:"+neighbourhoodsize+" Alpha:" + alpha + " threshold:" +threshold);
                    this.runDiffusion(neighbourhoodsize,alpha,threshold,holdoutFraction);
                }
            }
        }
    }

    public void runDiffusion(int neighbourhoodsize, double alpha, double thresholdFraction, Double holdoutFraction){
        LenskitConfiguration config = new LenskitConfiguration();
        config.bind(ItemScorer.class).to(itemscorer);
        config.bind(BaselineScorer.class, ItemScorer.class).to(baselinescorer);
        config.bind(UserMeanBaseline.class, ItemScorer.class).to(usermeanbaseline);
        config.bind(UserVectorNormalizer.class).to(uservectornormalizer);
        config.bind(NeighborFinder.class).to(neighborhoodfinder);
        config.bind(DiffusionModel.class).to(diffusionmodel);
        config.bind(similaritymatrixbuilder.getInterfaces()[0]).to(similaritymatrixbuilder);
        config.bind(UtilityMatrixNormalizer.class).to(utilitymatrixnormalizer);
        config.bind(LaplacianMatrixBuilder.class).to(laplacianbuilder);
        config.bind(VectorSimilarity.class).to(vectorsimilarity);
        config.set(NeighborhoodSize.class).to(neighbourhoodsize);
        config.set(MeanDamping.class).to(meanDamping);
        config.set(Alpha_nL.class).to(alpha);
        config.set(ThresholdFraction.class).to(thresholdFraction);


        AlgorithmInstance algo = new AlgorithmInstance("alg",config);

        Properties EvalProps = new Properties();
        EvalProps.setProperty(EvalConfig.THREAD_COUNT_PROPERTY, Integer.toString(5));
        SimpleEvaluator simpleEval = new SimpleEvaluator(EvalProps);
        simpleEval.addAlgorithm(algo);

        File in = new File(dataFileName);
        CSVDataSourceBuilder builder = new CSVDataSourceBuilder(in);
        builder.setDelimiter("\t");
        CSVDataSource dat = builder.build();

        //use 5-fold CV

        if (holdoutFraction == null){
            //use the default 10 ratings from each user for test set
            simpleEval.addDataset(dat,5);
        } else {
            //use holdout fraction ratings from each user
            simpleEval.addDataset(dat,5,holdoutFraction);
        }

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
        File out = new File("./" + itemscorer.getSimpleName()+ "/" + similaritymatrixbuilder.getSimpleName() +
                "/" + utilitymatrixnormalizer.getSimpleName() + "/" + laplacianbuilder.getSimpleName()
                + "/" + vectorsimilarity.getSimpleName() + "/knn_" + Integer.toString(neighbourhoodsize) + "_alphanL_" +
                Double.toString(alpha) + "_threshold_" + Double.toString(thresholdFraction) + ".csv");
        simpleEval.setOutput(out);

        try{
            simpleEval.call();
        } catch (Exception e){
            System.out.println(e.getMessage());
        }

    }


    public void runRegular(int neighbourhoodsize, Double holdoutFraction){
        LenskitConfiguration config = new LenskitConfiguration();
        config.bind(ItemScorer.class).to(itemscorer);
        config.bind(BaselineScorer.class, ItemScorer.class).to(baselinescorer);
        config.bind(UserMeanBaseline.class, ItemScorer.class).to(usermeanbaseline);
        config.bind(UserVectorNormalizer.class).to(uservectornormalizer);
        config.bind(NeighborFinder.class).to(neighborhoodfinder);
        config.bind(VectorSimilarity.class).to(regular_vectorsimilarity);
        config.set(NeighborhoodSize.class).to(neighbourhoodsize);
        config.set(MeanDamping.class).to(meanDamping);

        AlgorithmInstance algo = new AlgorithmInstance("alg",config);

        Properties EvalProps = new Properties();
        EvalProps.setProperty(EvalConfig.THREAD_COUNT_PROPERTY, Integer.toString(5));
        SimpleEvaluator simpleEval = new SimpleEvaluator(EvalProps);
        simpleEval.addAlgorithm(algo);

        File in = new File(dataFileName);
        CSVDataSourceBuilder builder = new CSVDataSourceBuilder(in);
        builder.setDelimiter("\t");
        CSVDataSource dat = builder.build();

        //use 5-fold CV

        if (holdoutFraction == null){
            //use the default 10 ratings from each user for test set
            simpleEval.addDataset(dat,5);
        } else {
            //use holdout fraction ratings from each user
            simpleEval.addDataset(dat,5,holdoutFraction);
        }

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
        File out = new File("./" + itemscorer.getSimpleName()+ "/" + regular_vectorsimilarity.getSimpleName() +
                "/knn_" + Integer.toString(neighbourhoodsize) + ".csv");
        simpleEval.setOutput(out);

        try{
            simpleEval.call();
        } catch (Exception e){
            System.out.println(e.getMessage());
        }

    }

}

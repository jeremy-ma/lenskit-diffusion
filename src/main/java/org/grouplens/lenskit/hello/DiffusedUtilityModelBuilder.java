package org.grouplens.lenskit.hello;

import com.jmatio.io.MatFileReader;
import com.jmatio.types.MLDouble;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.grouplens.lenskit.core.Transient;
import org.grouplens.lenskit.data.dao.EventDAO;

import javax.inject.Inject;
import javax.inject.Provider;
import java.io.File;
import java.lang.reflect.Field;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * by jeremy ma
 */
public class DiffusedUtilityModelBuilder implements Provider<DiffusedUtilityModel> {
    private final EventDAO dao;
    private final String diffusionMatrixFileName;
    private static final String precomputedFilePath = "precomputed_matrices/";

    @Inject
    public DiffusedUtilityModelBuilder(@Transient EventDAO dao, @MatrixFileName String diffusionFileName) {
        this.dao = dao;
        this.diffusionMatrixFileName = diffusionFileName;
    }

    @Override
    public DiffusedUtilityModel get() {

        // get the name of the input file
        //System.out.println(dao.toString());
        //System.out.println(dao.getClass());

        File trainingFile = null;
        //hacky way to determine which partition to get similarity from (reflection
        try{
            Field f = dao.getClass().getDeclaredField("inputFile");
            f.setAccessible(true);
            trainingFile = (File) f.get(dao);
        } catch (Exception e){
            System.out.println("Ahhhhh man!");
        }
        System.out.println(trainingFile.getName());
        Pattern pattern = Pattern.compile("train.(\\d).csv");
        Matcher matcher = pattern.matcher(trainingFile.getName());
        matcher.find();
        int partitionNum = Integer.parseInt(matcher.group(1));
        System.out.println(partitionNum);
        RealMatrix diffusedUtilityMatrix = null;

        //read in the precomputed matrix
        //TODO: do all preprocessing in Java, make similarity matrix etc.
        try{
            MatFileReader reader = new MatFileReader(precomputedFilePath + String.valueOf(partitionNum) + '_' + this.diffusionMatrixFileName);
            MLDouble red = (MLDouble) reader.getMLArray("utility");
            double [][] diffusion = red.getArray();
            diffusedUtilityMatrix = MatrixUtils.createRealMatrix(diffusion);
            System.out.println("Matrix is made");
        } catch (Exception e){
            System.out.println(e.getMessage());
            System.out.println("Failed to read in the diffused utility matrix");
        }

        return new DiffusedUtilityModel(diffusedUtilityMatrix);
    }
}

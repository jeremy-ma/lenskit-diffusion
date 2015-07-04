/*
 * LensKit, an open source recommender systems toolkit.
 * Copyright 2010-2014 LensKit Contributors.  See CONTRIBUTORS.md.
 * Work on LensKit has been funded by the National Science Foundation under
 * grants IIS 05-34939, 08-08692, 08-12148, and 10-17697.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation; either version 2.1 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program; if not, write to the Free Software Foundation, Inc., 51
 * Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 */
package org.grouplens.lenskit.vectors.similarity;

import com.jmatio.io.MatFileReader;
import com.jmatio.types.MLDouble;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.stat.correlation.PearsonsCorrelation;
import org.grouplens.lenskit.core.Shareable;
import org.grouplens.lenskit.vectors.SparseVector;
import org.grouplens.lenskit.vectors.VectorEntry;
import org.grouplens.lenskit.vectors.Vectors;

import javax.inject.Inject;
import java.io.Serializable;
import java.util.HashMap;

import static java.lang.Math.sqrt;

/**
 * Similarity function using Pearson correlation.
 *
 * <p>This class implements the Pearson correlation similarity function over
 * sparse vectors.  Only the items occurring in both vectors are considered when
 * computing the variance.
 *
 * <p>See Desrosiers, C. and Karypis, G., <i>A Comprehensive Survey of
 * Neighborhood-based Recommendation Methods</i>.  In Ricci, F., Rokach, L.,
 * Shapira, B., and Kantor, P. (eds.), <i>RecommenderEngine Systems Handbook</i>,
 * Springer. 2010, pp. 107-144.
 *
 * @author <a href="http://www.grouplens.org">GroupLens Research</a>
 */
@Shareable
public class DiffusedPearsonCorrelation implements VectorSimilarity, Serializable {
    private static final long serialVersionUID = 1L;
    private RealMatrix diffMatrix = null;
    private HashMap<SparseVector, ArrayRealVector> cache = null;
    private PearsonsCorrelation pearson = null;
    private final double shrinkage;

    public DiffusedPearsonCorrelation() {
        this(0);
    }

    @Inject
    public DiffusedPearsonCorrelation(@SimilarityDamping double s) {
        shrinkage = s;
        try{
            MatFileReader reader = new MatFileReader("ml100k_udiff.mat");
            MLDouble red = (MLDouble) reader.getMLArray("ml100k_udiff");
            double [][] diffusion = red.getArray();
            this.diffMatrix = MatrixUtils.createRealMatrix(diffusion);
            System.out.println("Matrix is made");
        } catch (Exception e){
            System.out.println(e.getMessage());
            System.out.println("Failed to read in the diffusion matrix");
        }
        //initialise the cache
        cache = new HashMap<SparseVector, ArrayRealVector>();


        pearson = new PearsonsCorrelation();

    }

    @Override
    public double similarity(SparseVector vec1, SparseVector vec2) {


        // First check for empty vectors - then we can assume at least one element
        if (vec1.isEmpty() || vec2.isEmpty()) {
            return 0;
        }

        ArrayRealVector v_diff;
        ArrayRealVector w_diff;

        if (( v_diff = this.cache.get(vec1) ) == null){
            v_diff = this.getDiffused(vec1);
            this.cache.put(vec1,v_diff);
        }

        if (( w_diff = this.cache.get(vec2) ) == null){
            w_diff = this.getDiffused(vec2);
            this.cache.put(vec2,w_diff);
        }


        double corr = pearson.correlation(v_diff.getDataRef(),w_diff.getDataRef());
        System.out.println(corr);
        return corr;

    }

    private ArrayRealVector getDiffused(SparseVector v){
        ArrayRealVector w = new ArrayRealVector(this.diffMatrix.getColumnDimension());

        for (VectorEntry entry:v){
            w.setEntry((int) entry.getKey() - 1, entry.getValue());
        }

        ArrayRealVector w_diff = (ArrayRealVector) this.diffMatrix.preMultiply(w);

        return w_diff;

    }


    @Override
    public boolean isSparse() {
        return true;
    }

    @Override
    public boolean isSymmetric() {
        return true;
    }

    @Override
    public String toString() {
        return String.format("DiffusedPearson[d=%s]", shrinkage);
    }
}

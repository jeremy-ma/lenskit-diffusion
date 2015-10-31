package org.grouplens.lenskit.diffusion.Iterative;

import it.unimi.dsi.fastutil.longs.LongSortedSet;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.grouplens.lenskit.ItemScorer;
import org.grouplens.lenskit.basic.AbstractItemScorer;
import org.grouplens.lenskit.data.dao.UserEventDAO;
import org.grouplens.lenskit.data.event.Event;
import org.grouplens.lenskit.data.history.History;
import org.grouplens.lenskit.data.history.UserHistory;
import org.grouplens.lenskit.data.history.UserHistorySummarizer;
import org.grouplens.lenskit.diffusion.UserCF.UserCFDiffusionModel;
import org.grouplens.lenskit.diffusion.general.DiffusionModel;
import org.grouplens.lenskit.diffusion.general.VectorUtils;
import org.grouplens.lenskit.knn.item.ItemScoreAlgorithm;
import org.grouplens.lenskit.knn.item.NeighborhoodScorer;
import org.grouplens.lenskit.knn.item.model.ItemItemModel;
import org.grouplens.lenskit.symbols.Symbol;
import org.grouplens.lenskit.transform.normalize.UserVectorNormalizer;
import org.grouplens.lenskit.transform.normalize.VectorTransformation;
import org.grouplens.lenskit.vectors.MutableSparseVector;
import org.grouplens.lenskit.vectors.SparseVector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.inject.Inject;
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

import org.grouplens.lenskit.ItemScorer;
import org.grouplens.lenskit.basic.AbstractItemScorer;
import org.grouplens.lenskit.data.dao.UserEventDAO;
import org.grouplens.lenskit.data.event.Event;
import org.grouplens.lenskit.data.history.History;
import org.grouplens.lenskit.data.history.UserHistory;
import org.grouplens.lenskit.data.history.UserHistorySummarizer;
import org.grouplens.lenskit.knn.item.ItemScoreAlgorithm;
import org.grouplens.lenskit.knn.item.NeighborhoodScorer;
import org.grouplens.lenskit.knn.item.model.ItemItemModel;
import org.grouplens.lenskit.symbols.Symbol;
import org.grouplens.lenskit.transform.normalize.UserVectorNormalizer;
import org.grouplens.lenskit.transform.normalize.VectorTransformation;
import org.grouplens.lenskit.vectors.MutableSparseVector;
import org.grouplens.lenskit.vectors.SparseVector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.inject.Inject;

/**
 * Score items using an item-item CF model. User ratings are <b>not</b> supplied
 * as default preferences.
 *
 * @author <a href="http://www.grouplens.org">GroupLens Research</a>
 */
public class IterativeDiffusionItemScorer extends AbstractItemScorer {
    private static final Logger logger = LoggerFactory.getLogger(IterativeDiffusionItemScorer.class);



    private final UserEventDAO dao;
    @Nonnull
    protected final UserVectorNormalizer normalizer;
    protected final UserHistorySummarizer summarizer;
    private RealMatrix diffusionMatrix;

    /**
     * Construct a new item-item scorer.
     *
     * @param dao    The DAO.
     * @param sum    The history summarizer.
     */
    @Inject
    public IterativeDiffusionItemScorer(UserEventDAO dao,
                              UserHistorySummarizer sum,
                              UserVectorNormalizer norm,
                              UserCFDiffusionModel diffusionModel) {
        this.dao = dao;
        summarizer = sum;
        normalizer = norm;
        diffusionMatrix = diffusionModel.getDiffusionMatrix();
        logger.debug("configured IterativeDiffusionItemScorer");
    }

    /**
     * Score items by computing predicted ratings.
     *
     * @see ItemScoreAlgorithm#scoreItems(ItemItemModel, org.grouplens.lenskit.vectors.SparseVector, org.grouplens.lenskit.vectors.MutableSparseVector, NeighborhoodScorer)
     */
    @Override
    public void score(long user, @Nonnull MutableSparseVector scores) {
        UserHistory<? extends Event> history = dao.getEventsForUser(user, summarizer.eventTypeWanted());
        if (history == null) {
            history = History.forUser(user);
        }
        SparseVector summary = summarizer.summarize(history);
        VectorTransformation transform = normalizer.makeTransformation(user, summary);
        MutableSparseVector normed = summary.mutableCopy();
        transform.apply(normed);
        scores.clear();
        int numItems = 1682;
        //algorithm.scoreItems(model, normed, scores, scorer);
        int num_updates = 300;
        double update_rate = 1;
        double threshold = 0.01;
        RealVector z_out = diffusionMatrix.preMultiply(VectorUtils.toRealVector(numItems,normed));
        boolean updated = true;
        LongSortedSet known = normed.keySet();
        int count_iter = 0;
        for (int i=0; i<num_updates && updated; i++){
            updated = false;
            RealVector temp = diffusionMatrix.preMultiply(z_out);
            temp.mapMultiplyToSelf(z_out.getNorm() / temp.getNorm());
            RealVector temp_diff = z_out.add(temp.mapMultiplyToSelf(-1.0));
            for (int j=0;j<numItems;j++) {
                if (!known.contains((long) (j + 1))) {
                    //if the rating is not one of the known ones
                    if (Math.abs(temp_diff.getEntry(j)) > threshold) {
                        // if difference is large enough, update
                        updated = true;
                        z_out.setEntry(j, (1.0 - update_rate) * z_out.getEntry(j) + update_rate * temp.getEntry(j));
                    }
                }
            }
            count_iter++;
        }
        System.out.println(count_iter);
        LongSortedSet testDomain = scores.keyDomain();
        //fill up the score vector
        for (int i=0; i<numItems;i++){
            if (testDomain.contains((long) (i+1))){
                scores.set((long)(i+1), z_out.getEntry(i));
            }
        }

        // untransform the scores
        transform.unapply(scores);
        System.out.println(scores);
    }
}

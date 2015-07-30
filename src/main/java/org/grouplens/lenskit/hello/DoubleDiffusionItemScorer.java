package org.grouplens.lenskit.hello;

import it.unimi.dsi.fastutil.longs.Long2ObjectMap;
import it.unimi.dsi.fastutil.longs.Long2ObjectOpenHashMap;
import org.apache.commons.lang3.mutable.Mutable;
import org.apache.commons.math3.linear.RealMatrix;
import org.grouplens.lenskit.ItemScorer;
import org.grouplens.lenskit.basic.AbstractItemScorer;
import org.grouplens.lenskit.collections.LongKeyDomain;
import org.grouplens.lenskit.vectors.ImmutableSparseVector;
import org.grouplens.lenskit.vectors.MutableSparseVector;
import org.grouplens.lenskit.vectors.SparseVector;

import javax.annotation.Nonnull;
import javax.inject.Inject;
import java.util.ArrayList;
import java.util.Collection;

/**
 * Created by jeremyma on 28/07/15.
 */
public class DoubleDiffusionItemScorer extends AbstractItemScorer {

    private static final long serialVersionUID = 1L;

    private final Long2ObjectMap<ImmutableSparseVector> userData;

    @Inject
    private DoubleDiffusionItemScorer(DiffusedUtilityModel model) {
        RealMatrix complete_utility_matrix = model.getDiffusionMatrix();
        int numUsers = complete_utility_matrix.getRowDimension();
        int numItems = complete_utility_matrix.getColumnDimension();
        double[][] complete_utility_array = complete_utility_matrix.getData();
        ArrayList<Long> key_domain = new ArrayList<Long>(numItems);
        for (int i=0; i<numItems; i++){
            key_domain.add(i, new Long(i+1));
        }

        //expected load factor is 1
        userData = new Long2ObjectOpenHashMap<ImmutableSparseVector>(numUsers, 1F);

        LongKeyDomain item_ids = LongKeyDomain.fromCollection(key_domain, true);

        for (int i=0; i<complete_utility_array.length; i++){
            MutableSparseVector user_ratings = MutableSparseVector.create(key_domain);
            //TODO: do this without a loop
            for (int j=0; j<complete_utility_array[i].length; j++){
                user_ratings.set((long) j+1, complete_utility_array[i][j]);
            }
            userData.put(i+1, user_ratings.immutable());
        }
    }

    @Override
    public void score(long user, @Nonnull MutableSparseVector scores) {
        SparseVector sv = userData.get(user);
        scores.clear();
        if (sv != null) {
            scores.set(sv);
        }
    }

}

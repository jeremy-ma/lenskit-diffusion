import numpy as np
from scipy.stats import pearsonr
import scipy.io
from scipy.sparse import csgraph
from scipy.spatial.distance import cosine
import sys, editdistance


defaultNumUsers,defaultNumItems = 943, 1682


#create a utility matrix from a ratings file
#rows correspond to users, items (movies) corresponds to columns
def create_utility_matrix(filename, numUsers, numItems):

    ratings_f = open(filename, 'r')
    utility = np.zeros((numUsers,numItems))
    for line in ratings_f:
        line = line.rstrip()
        fields = line.split('\t')
        uid = int(fields[0]) - 1
        movid = int(fields[1]) - 1
        rating = int(fields[2])
        utility[uid][movid] = rating

    return utility

def symmetrize(a):
    return a + a.T - np.diag(a.diagonal())

# create similarity matrix based on the pearson correlation measure
def create_similarity_pearson_correlation(utility, numUsers, numItems):

    similarity = np.zeros((numItems,numItems))

    for i in xrange(numItems):
        for j in xrange(i,numItems):
            #calculate the pearson correlation coefficient between two movies
            #print pearsonr(utility[:,i],utility[:,j])
            if np.count_nonzero(utility[:,i]) == 0 or np.count_nonzero(utility[:,j]) == 0:
                continue
            corr, pval = pearsonr(utility[:,i],utility[:,j])
            similarity[i][j], similarity[j][i] = corr, corr

    return similarity


def create_similarity_cosine(utility, numUsers, numItems):
    # create similarity matrix using cosine similarity
    similarity = np.zeros((numItems,numItems))

    for i in xrange(numItems):
        for j in xrange(i,numItems):
            if np.count_nonzero(utility[:,i]) == 0 or np.count_nonzero(utility[:,j]) == 0:
                continue
            sim = 1 - cosine(utility[:,i],utility[:,j])
            similarity[i][j], similarity[j][i] = sim, sim

    return similarity

def create_similarity_adjusted_cosine(utility,numUsers,numItems):
    # create similarity matrix using the adjusted cosine similarity measure
    similarity = np.zeros((numItems,numItems))

    # mean centre the ratings for each user (leaving 0's unaffected)

    for u in xrange(numUsers):
        zero_indices = np.where(utility[u,:] == 0)[0]
        utility[u,:] = utility[u,:] - utility[u,:].mean()
        utility[u,:][zero_indices] = 0.0

    for i in xrange(numItems):
        for j in xrange(i,numItems):
            #calculate the pearson correlation coefficient between two movies
            #print pearsonr(utility[:,i],utility[:,j])
            if np.count_nonzero(utility[:,i]) == 0 or np.count_nonzero(utility[:,j]) == 0:
                continue
            sim = 1 - cosine(utility[:,i],utility[:,j])
            similarity[i][j], similarity[j][i] = sim, sim

    return similarity

def transformation(similarity):
    return similarity

def create_diffusion(simmat, alpha_nL=1.0, threshold_fraction=0.3):
    # performs the calculations to create the diffusion matrices

    #thresholding code for absolute laplacian
    """
    abs_threshold = find_threshold(np.abs(simmat), threshold_fraction)
    copy = np.copy(simmat)
    copy[np.logical_and(copy > -abs_threshold, copy < abs_threshold)] = 0.0
    # print abs_threshold
    print "{0} percent nonzero (absolute)".format(np.count_nonzero(copy)/float(len(simmat)**2) * 100.0 )
    """
    copy = simmat

    # calculate absolute laplacian
    L_abs = np.zeros(copy.shape)
    for i in xrange(len(copy)):
        L_abs[i][i] = np.sum(np.abs(copy[i,:]))
    L_abs = L_abs - copy

    # apply thresholding which reduces number of edges to the desired fraction for
    threshold = find_threshold(simmat, threshold_fraction)
    simmat[simmat < threshold] = 0.0
    # print threshold
    print "{0} percent nonzero".format(np.count_nonzero(simmat)/float(len(simmat)**2) * 100.0 )

    ##################################################
    # apply transformation function x^2
    simmat = simmat ** 2
    ##################################################

    L = csgraph.laplacian(simmat, normed=False)
    L_n = csgraph.laplacian(simmat, normed=True)

    # calculate diffusion rates
    ratio_diagL_diagNL = L.diagonal().sum() / L_n.diagonal().sum()
    alpha_L = alpha_nL / ratio_diagL_diagNL
    #alpha_L 5.5377e-07

    diff = np.linalg.inv(np.eye(len(simmat)) + alpha_L * L)
    diff_n = np.linalg.inv(np.eye(len(simmat)) + alpha_nL * L_n)
    diff_abs = np.linalg.inv(np.eye(len(simmat)) + alpha_L * L_abs)

    return (diff, diff_n, diff_abs)

def find_threshold(similarity, threshold_fraction, numItems = defaultNumItems):
    # binary search the correct threshold

    num_iters = 15;

    copy = similarity.copy()

    lo = 0.0
    hi = 1.0
    np.count_nonzero(similarity)

    for i in xrange(num_iters):
        mid = (lo + hi) / 2.0
        copy[copy < mid] = 0.0
        percent = np.count_nonzero(copy)/float(numItems**2)
        if percent > threshold_fraction:
            # raise the threshold
            lo = mid
        else:
            hi = mid
        copy = similarity.copy()
        #print (lo,hi)

    return mid

def main_similarity(sim_func ='cosine', filename='ml-100k/u.data', numUsers=defaultNumUsers, numItems=defaultNumItems):
    # writes similarity matrices to file

    print "generating utility matrix......"
    utility = create_utility_matrix(filename, numUsers,numItems)
    print "done"

    print "creating similarity matrix....."
    if sim_func == 'cosine':
        similarity = create_similarity_cosine(utility, numUsers, numItems)
    elif sim_func == 'adjusted_cosine':
        similarity = create_similarity_adjusted_cosine(utility, numUsers, numItems)
    elif sim_func == 'pearson':
        similarity = create_similarity_pearson_correlation(utility, numUsers,numItems)

    scipy.io.savemat(sim_func + '_similarity_ml100k.mat', mdict={'similarity': similarity})

    print "done"

def main_diffusion(sim_func = 'cosine', alpha_nL=1.0, threshold_fraction = 0.3, numItems=defaultNumItems, numUsers=defaultNumUsers):
    # this function creates and writes the diffusion matrix files

    # use a similarity matrix which has nonzero values
    similarity = scipy.io.loadmat(sim_func + '_similarity_ml100k.mat')['similarity']

    print "creating diffusion matrix alpha_nL: " + str(alpha_nL)
    diff, diff_n, diff_abs = create_diffusion(similarity, alpha_nL,threshold_fraction)
    print "saving"

    # save both the normalized and regular diffusion matrices
    scipy.io.savemat('ml100k_udiff.mat',mdict={'diffusion':diff})
    scipy.io.savemat('ml100k_udiff_n.mat',mdict={'diffusion':diff_n})
    scipy.io.savemat('ml100k_udiff_abs.mat',mdict={'diffusion':diff_abs})


# removes the year and foreign language translation from the movie name
def process_name(name):
    return name.rstrip()


def isolate_name(name):
    return name.split('(')[0].rstrip()


def modify_ml100k_movienames():
    # reduce ml10M dataset to just the movies in the ml100k dataset
    ml100k_mapping = {}
    ml100k_movies = open('ml-100k/u.item.modified')
    ml10M_movies = open('../preprocessing/ml-1m/movies.dat')
    for line in ml100k_movies:
        line = line.split('|')
        movid = int(line[0])
        movname = process_name(line[1])
        ml100k_mapping[movname] = movid

    ml10M_to_ml100k = {}

    ml10M_mapping = {}

    num_matched = 0
    for line in ml10M_movies:
        line = line.split('::')
        movname = process_name(line[1])
        ml10M_mapping[movname] = int(line[0])
        if movname in ml100k_mapping:
            ml10M_to_ml100k[int(line[0])] = ml100k_mapping[movname]
            num_matched += 1

    print num_matched

    ml100k_movies = open('ml-100k/u.item.modified')

    new = open('ml-100k/u.item.modified.twice', 'w')
    stuff = []
    replacements = set()

    for line in ml100k_movies:
        line = line.split('|')
        movname = process_name(line[1])
        if movname not in ml10M_mapping:
            print movname
            replaced = False
            min_edit_dist = 1000000000000
            movieList = open('../preprocessing/ml-1m/movies.dat')

            for movie in movieList:

                movie = movie.split('::')[1]
                if process_name(movie) in ml100k_mapping or movie in replacements:
                    continue

                if movname in movie or isolate_name(movname) in movie:
                    print "replacing {0} with {1}".format(movname, movie)
                    line[1] = movie
                    replaced = True
                    replacements.add(movie)
                    break
                else:
                    dist = editdistance.eval(movie, movname)
                    if dist < min_edit_dist:
                        min_edit_dist = dist
                        candidate = movie

            if replaced is False:
                print "Replacing Min Edit Distance {0} with {1}".format(movname, candidate)
                line[1] = candidate
                replacements.add(candidate)

            print "######################"

        line = '|'.join(line)
        stuff.append(line)

    new.write(''.join(stuff))


def reduce_ml10M_ml100k():
    # reduce ml10M dataset to just the movies in the ml100k dataset
    ml100k_mapping = {}
    ml100k_movies = open('ml-100k/u.item.modified.twice')
    ml10M_movies = open('../preprocessing/ml-1m/movies.dat')
    for line in ml100k_movies:
        line = line.split('|')
        movid = int(line[0])
        movname = process_name(line[1])
        if movname in ml100k_mapping:
            print movname
        ml100k_mapping[movname] = movid


    ml10M_to_ml100k = {}

    ml10M_mapping = {}
    matched = set()
    num_matched = 0
    for line in ml10M_movies:
        line = line.split('::')
        movname = process_name(line[1])
        ml10M_mapping[movname] = int(line[0])
        if movname in ml100k_mapping:
            if movname in matched:
                print movname
            ml10M_to_ml100k[int(line[0])] = ml100k_mapping[movname]
            num_matched += 1
            matched.add(movname)

    print num_matched, len(ml100k_mapping)
    new = open('ml10M_reduced/ml1M_ml100K.dat', 'a')
    ml10M_data = open('../preprocessing/ml-1m/ratings.dat')

    for line in ml10M_data:
        line = line.split('::')
        ml10M_movid = int(line[1])
        if ml10M_movid in ml10M_to_ml100k:
            line[1] = str(ml10M_to_ml100k[ml10M_movid])
            new.write('\t'.join(line))



if __name__=='__main__':

    reduce_ml10M_ml100k()












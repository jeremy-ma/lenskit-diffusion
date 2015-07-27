import numpy as np
from scipy.stats import pearsonr
import scipy.io
from scipy.sparse import csgraph
from scipy.spatial.distance import cosine
import sys, editdistance
import pdb

defaultNumUsers,defaultNumItems = 943, 1682
matrix_path = 'precomputed_matrices/'

#create a utility matrix from a ratings file
#rows correspond to users, items (movies) corresponds to columns
def create_utility_matrix(filename, numUsers, numItems, file_delimiter='\t'):

    ratings_f = open(filename, 'r')
    utility = np.zeros((numUsers,numItems))
    for line in ratings_f:
        line = line.rstrip()
        fields = line.split(file_delimiter)
        uid = int(fields[0]) - 1
        movid = int(fields[1]) - 1
        rating = float(fields[2])
        utility[uid][movid] = rating

    return utility

def symmetrize(a):
    return a + a.T - np.diag(a.diagonal())

# create similarity matrix based on the pearson correlation measure
def create_similarity_pearson_correlation(utility):
    numColumns = utility.shape[1]
    similarity = np.zeros((numColumns,numColumns))
    for i in xrange(numColumns):
        for j in xrange(i,numColumns):
            #calculate the pearson correlation coefficient between two movies
            #print pearsonr(utility[:,i],utility[:,j])
            if np.count_nonzero(utility[:,i]) == 0 or np.count_nonzero(utility[:,j]) == 0:
                continue
            corr, pval = pearsonr(utility[:,i],utility[:,j])
            similarity[i][j], similarity[j][i] = corr, corr

    return similarity


def create_similarity_cosine(utility):
    # create similarity matrix using cosine similarity
    numColumns = utility.shape[1]
    similarity = np.zeros((numColumns,numColumns))
    for i in xrange(numColumns):
        for j in xrange(i,numColumns):
            if np.count_nonzero(utility[:,i]) == 0 or np.count_nonzero(utility[:,j]) == 0:
                continue
            sim = 1 - cosine(utility[:,i],utility[:,j])
            similarity[i][j], similarity[j][i] = sim, sim

    return similarity

def create_similarity_adjusted_cosine(utility):
    # create similarity matrix using the adjusted cosine similarity measure
    numRows, numColumns = utility.shape
    similarity = np.zeros((numColumns,numColumns))
    # mean centre the ratings for each user (leaving 0's unaffected)
    for u in xrange(numRows):
        if np.count_nonzero(utility[u,:]) == 0:
            # don't mean center zero vectors
            continue
        zero_indices = np.where(utility[u,:] == 0)[0]
        utility[u,:] = utility[u,:] - utility[u,:].sum() / float(np.count_nonzero(utility[u,:]))
        utility[u,:][zero_indices] = 0.0

    for i in xrange(numColumns):
        for j in xrange(i,numColumns):
            #calculate the cosine correlation between two movies
            if np.count_nonzero(utility[:,i]) == 0 or np.count_nonzero(utility[:,j]) == 0:
                continue
            sim = 1 - cosine(utility[:,i],utility[:,j])
            similarity[i][j], similarity[j][i] = sim, sim

    return similarity

def transformation_linear(similarity):
    # no transform
    return similarity

def transformation_squared(similarity):
    return similarity ** 2

def transformation_cubed(similarity):
    return similarity ** 3

def transformation_exp(similarity):
    return np.exp(similarity) - 1.0


def create_diffusion(simmat, alpha_nL=1.0, threshold_fraction=0.3, transform=transformation_linear, threshold_absolute=True):
    # performs the calculations to create the diffusion matrices:

    #thresholding code for absolute laplacian
    """
    abs_threshold = find_threshold(np.abs(simmat), threshold_fraction)
    copy = np.copy(simmat)
    copy[np.logical_and(copy > -abs_threshold, copy < abs_threshold)] = 0.0
    # print abs_threshold
    print "{0} percent nonzero (absolute)".format(np.count_nonzero(copy)/float(len(simmat)**2) * 100.0 )
    """
    copy = np.copy(simmat)
    if threshold_absolute is True:
        print "Thresholding for absolute laplacian"
        threshold = find_threshold(np.abs(copy), threshold_fraction)
        copy[np.logical_and(copy > -(threshold)/2, copy<threshold)] = 0.0
        print "{0} percent nonzero".format(np.count_nonzero(copy)/float(len(simmat)**2) * 100.0 )
        print "{0} percent negative".format((copy<0).sum()/float(len(simmat)**2))

    # calculate absolute laplacian
    L_abs = np.zeros(copy.shape)
    for i in xrange(len(copy)):
        L_abs[i][i] = np.sum(np.abs(copy[i,:]))
    L_abs = L_abs - copy

    # make absolute normalized laplacian

    D = np.eye(copy.shape[0]) * np.diag(L_abs)
    D_inv = 1.0 / D
    D_inv[np.isinf(D_inv)] = 0.0
    L_abs_n = np.dot(D_inv, L_abs)
    # pdb.set_trace()

    # apply thresholding which reduces number of edges to the desired fraction for
    threshold = find_threshold(simmat, threshold_fraction)
    simmat[simmat < threshold] = 0.0
    # print threshold
    print "{0} percent nonzero".format(np.count_nonzero(simmat)/float(len(simmat)**2) * 100.0 )

    ##################################################
    # apply transformation function
    simmat = transform(simmat)
    ##################################################

    L = csgraph.laplacian(simmat, normed=False)
    L_n = csgraph.laplacian(simmat, normed=True)

    # calculate diffusion rates
    ratio_diagL_diagNL = L.diagonal().sum() / L_n.diagonal().sum()
    alpha_L = alpha_nL / ratio_diagL_diagNL
    ratio_diagabsL_diagNL = L_abs.diagonal().sum() / L_n.diagonal().sum()
    alpha_absL = alpha_nL / ratio_diagabsL_diagNL

    diff = np.linalg.inv(np.eye(len(simmat)) + alpha_L * L)
    diff_n = np.linalg.inv(np.eye(len(simmat)) + alpha_nL * L_n)
    diff_abs = np.linalg.inv(np.eye(len(simmat)) + alpha_absL * L_abs)
    diff_abs_n = np.linalg.inv(np.eye(len(simmat)) + alpha_nL * L_abs_n)

    return (diff, diff_n, diff_abs, diff_abs_n)

def find_threshold(similarity, threshold_fraction, numItems = defaultNumItems):
    # binary search the correct threshold

    num_iters = 15;

    copy = similarity.copy()

    lo = 0.0
    hi = 1.0

    for i in xrange(num_iters):
        mid = (lo + hi) / 2.0
        copy[copy < mid] = 0.0
        percent = np.count_nonzero(copy)/float(similarity.shape[0]**2)
        # print percent
        if percent > threshold_fraction:
            # raise the threshold
            lo = mid
        else:
            hi = mid
        copy = similarity.copy()

    return mid

def main_similarity(sim_func ='cosine', filename='ml-100k/u.data', 
                    numUsers=defaultNumUsers, numItems=defaultNumItems, 
                    outsuffix='_similarity_ml100k.mat', file_delimiter='\t', useruser=False):
    # writes similarity matrices to file

    print "generating utility matrix......"
    utility = create_utility_matrix(filename, numUsers, numItems, file_delimiter=file_delimiter)
    print "done"

    if useruser is True:
        utility = utility.T

    print "creating similarity matrix....."
    if sim_func == 'cosine':
        similarity = create_similarity_cosine(utility)
    elif sim_func == 'adjusted_cosine':
        similarity = create_similarity_adjusted_cosine(utility)
    elif sim_func == 'pearson':
        similarity = create_similarity_pearson_correlation(utility)

    fname = sim_func + outsuffix

    scipy.io.savemat(fname, mdict={'similarity': similarity})

    print "done"

def main_diffusion(sim_func = 'cosine', alpha_nL=1.0, threshold_fraction = 0.3, 
                    numItems=defaultNumItems, numUsers=defaultNumUsers, transform=transformation_linear,
                    simmat_suffix='_similarity_ml100k.mat', output_prefix=''):
    # this function creates and writes the diffusion matrix files

    # use a similarity matrix which has nonzero values
    similarity = scipy.io.loadmat(matrix_path + sim_func + simmat_suffix)['similarity']

    print "creating diffusion matrix alpha_nL: " + str(alpha_nL)
    diff, diff_n, diff_abs, diff_abs_n = create_diffusion(similarity, alpha_nL,threshold_fraction, transform)
    print "saving"

    # save both the normalized and regular diffusion matrices
    scipy.io.savemat(matrix_path + output_prefix + 'ml100k_udiff.mat',mdict={'diffusion':diff})
    scipy.io.savemat(matrix_path + output_prefix + 'ml100k_udiff_n.mat',mdict={'diffusion':diff_n})
    scipy.io.savemat(matrix_path + output_prefix + 'ml100k_udiff_abs.mat',mdict={'diffusion':diff_abs})
    scipy.io.savemat(matrix_path + output_prefix + 'ml100k_udiff_abs_n.mat', mdict={'diffusion':diff_abs_n})

def main_double_diffusion(input_file='ml-100k/u.data', sim_func='adjusted_cosine', alpha_nL=1.0, threshold_fraction=0.08,
                          transform=transformation_linear, output_prefix=''):
    
    # get utility matrix
    utility = create_utility_matrix(input_file, defaultNumUsers, defaultNumItems, file_delimiter=',')
    # create similarity matrices
    print "creating similarity matrix....."
    if sim_func == 'cosine':
        similarity_func = create_similarity_cosine
    elif sim_func == 'adjusted_cosine':
        similarity_func = create_similarity_adjusted_cosine
    elif sim_func == 'pearson':
        similarity_func = create_similarity_pearson_correlation



    similarity_uu = similarity_func(utility)
    similarity_ii = similarity_func(utility.T)

    print "creating diffusion matrices"

    diff_uu,diff_uu_n, diff_uu_abs, diff_uu_abs_n = create_diffusion(similarity_uu, alpha_nL, threshold_fraction,
                                                                     transform, threshold_absolute=True)
    diff_ii,diff_ii_n, diff_ii_abs, diff_ii_abs_n = create_diffusion(similarity_ii, alpha_nL, threshold_fraction,
                                                                     transform, threshold_absolute=True)

    # mean center utility matrix
    for i in xrange(utility.shape[0]):
        utility[i,:] = utility[i,:] - utility[i,:].sum() / float(np.count_nonzero(utility[i,:]))

    print utility.shape, diff_uu.shape

    print "diffusing utility matrices"
    util_diff = utility.dot(diff_uu)
    util_diff_n = utility.dot(diff_uu_n)

    print util_diff.shape, diff_ii.shape

    util_diff = diff_ii.dot(util_diff)
    util_diff_n = diff_ii_n.dot(util_diff)

    print "saving"

    scipy.io.savemat(matrix_path + output_prefix + 'ml100k_util_diff.mat',mdict={'utility':util_diff})
    scipy.io.savemat(matrix_path + output_prefix + 'ml100k_util_diff_n.mat',mdict={'utility':util_diff_n})

    return (util_diff,util_diff_n)

if __name__=='__main__':

    main_similarity()












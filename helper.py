import numpy as np
from scipy.stats import pearsonr
import scipy.io
from scipy.sparse import csgraph
from scipy.spatial.distance import cosine, pdist, squareform
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

def mean_center(utility, include_mean_vector=False):
    # mean center and zero entries
    zero_indices = np.where(utility==0)
    # mean centre the ratings for each user (leaving 0's unaffected)
    means = utility.sum(1)/(utility != 0).sum(1)
    means[np.isnan(means)] = 0.0
    means = means.reshape(-1,1)
    utility = utility - means
    utility[zero_indices] = 0.0

    if include_mean_vector is False:
        return utility
    else:
        return (utility, means)

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
    similarity = squareform(pdist(utility,'cosine'))
    similarity = 1 - similarity
    #set nan to zero
    similarity[np.isnan(similarity)] = 0.0

    return similarity

def create_similarity_adjusted_cosine(utility):
    # create similarity matrix using the adjusted cosine similarity measure
    utility = mean_center(utility)
    return create_similarity_cosine(utility)

def transformation_linear(similarity):
    # no transform
    return similarity

def transformation_squared(similarity):
    return similarity ** 2

def transformation_cubed(similarity):
    return similarity ** 3

def transformation_exp(similarity):
    return np.exp(similarity) - 1.0


def diffusion(simmat, alpha_nL=1.0, threshold_fraction=0.3, transform=transformation_linear, threshold_absolute=True):
    # performs the calculations to create the diffusion matrices:

    #thresholding code for absolute laplacian

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

def create_diffusion(sim_func = 'cosine', alpha_nL=1.0, threshold_fraction = 0.3, 
                    numItems=defaultNumItems, numUsers=defaultNumUsers, 
                    transform=transformation_linear, output_prefix=''):
    # this function creates and writes the diffusion matrix files

    # use a similarity matrix which has nonzero values
    similarity = scipy.io.loadmat(matrix_path + sim_func + simmat_suffix)['similarity']

    print "creating diffusion matrix alpha_nL: " + str(alpha_nL)
    diff, diff_n, diff_abs, diff_abs_n = diffusion(similarity, alpha_nL,threshold_fraction, transform)
    print "saving"

    return (diff, diff_n, diff_abs, diff_abs_n)

def main_double_diffusion(input_file='ml-100k/u.data', sim_func='adjusted_cosine', alpha_nL=1.0, threshold_fraction=0.08,
                          transform=transformation_linear, output_prefix='', reverse=False, order_dependent=True):
    
    if sim_func == 'cosine':
        similarity_func = create_similarity_cosine
    elif sim_func == 'adjusted_cosine':
        similarity_func = create_similarity_adjusted_cosine
    elif sim_func == 'pearson':
        similarity_func = create_similarity_pearson_correlation

    # default, diffuse by useruser diffusion then itemitem diffusion

    utility = create_utility_matrix(input_file, defaultNumUsers, defaultNumItems, file_delimiter=',')
    original_util = utility.copy()
    original_nonzero = np.nonzero(original_util)

    if reverse is True:
        utility = utility.T

    similarity_uu = similarity_func(utility)

    # print similarity_uu.shape
    diff_uu,diff_uu_n,_,_ = diffusion(similarity_uu, alpha_nL, threshold_fraction,
                                                                     transform, threshold_absolute=False)
    utility, mean = mean_center(utility, include_mean_vector=True)
    util_diff = diff_uu.dot(utility)
    util_diff_n = diff_uu_n.dot(utility)

    util_diff += mean
    util_diff_n += mean

    if order_dependent is True:
        similarity_ii = similarity_func(util_diff.T)
        similarity_ii_n = similarity_func(util_diff_n.T)
    else:
        #use originial similarity matrices
        if util_diff.shape != original_util.shape:
            similarity_ii = similarity_func(original_util)
        else:
            similarity_ii = similarity_func(original_util.T)
        similarity_ii_n = similarity_ii.copy()


    diff_ii,_,_,_= diffusion(similarity_ii, alpha_nL, threshold_fraction, transform, threshold_absolute=False)
    _,diff_ii_n,_,_ = diffusion(similarity_ii_n, alpha_nL, threshold_fraction, transform, threshold_absolute=False)

    util_diff, mean_diff = mean_center(util_diff.T, include_mean_vector=True)
    util_diff_n, mean_diff_n = mean_center(util_diff_n.T, include_mean_vector=True)

    # diffuse again then change back to previous form
    util_diff = diff_ii.dot(util_diff)
    util_diff_n = diff_ii_n.dot(util_diff_n)
    util_diff_complete_n = (util_diff_n + mean_diff_n)
    util_diff_complete = (util_diff + mean_diff)

    if reverse is False:
        util_diff = util_diff.T
        util_diff_n = util_diff_n.T
        util_diff_complete = util_diff_complete.T
        util_diff_complete_n = util_diff_complete_n.T

    #keep the original values for the diffused complete matrix
    util_diff_complete[original_nonzero] = original_util[original_nonzero]
    util_diff_complete_n[original_nonzero] = original_util[original_nonzero]

    print "saving"

    scipy.io.savemat(matrix_path + output_prefix + 'ml100k_util_diff.mat',mdict={'utility':util_diff})
    scipy.io.savemat(matrix_path + output_prefix + 'ml100k_util_diff_n.mat',mdict={'utility':util_diff_n})
    scipy.io.savemat(matrix_path + output_prefix + 'ml100k_util_diff_complete.mat', mdict={'utility':util_diff_complete})
    scipy.io.savemat(matrix_path + output_prefix + 'ml100k_util_diff_n_complete', mdict={'utility':util_diff_complete_n})

    return (util_diff,util_diff_n,util_diff_complete, util_diff_complete_n)

if __name__=='__main__':

    main_similarity()












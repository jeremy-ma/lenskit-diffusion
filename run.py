# some_file.py
import sys
import helper
import subprocess
import scipy.io
import numpy as np
import csv, pprint
from collections import defaultdict
from itertools import chain


run_command = "java -jar ./build/classes/artifacts/lenskit_algorithm_example_master_jar2/lenskit-algorithm-example-master.jar"

filename = 'ml-100k/u.data'
similarity_matrix_funcs = ['adjusted_cosine']
vector_similarity_funcs = ['cosine']
# alpha_nl_array = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
alpha_nl_array = [0.1, 0.2, 0.4, 0.8, 1.0, 2.0, 4.0, 8.0, 100.0]

partitions = 5
metrics = ['RMSE.ByUser','RMSE.ByRating','MAE.ByUser','MAE.ByRating']

# run test set
def run_test_set(threshold_fraction=0.3, transform=helper.transformation_linear):
    for vectorfunc in vector_similarity_funcs:
        for matrixfunc in similarity_matrix_funcs:
            for alpha_nL in alpha_nl_array:
                helper.main_diffusion(matrixfunc, alpha_nL, threshold_fraction, transform)
                dataFileName = "ml-100k/u.data"
                resultFileName = "results_" + vectorfunc + "vectorsim" + "_" \
                            + matrixfunc + "similaritymatrix_" + "alpha_nL_" + str(alpha_nL) + ".csv"
                args = " {0} {1} {2}".format(dataFileName, resultFileName, vectorfunc)
                subprocess.call(run_command + args, shell=True)

# run test set but make diffusion matrices for each partition
def run_test_set_partitions(threshold_fraction=0.3, transform=helper.transformation_linear, knn=40, useruser=False):
    for vectorfunc in vector_similarity_funcs:
        for matrixfunc in similarity_matrix_funcs:
            for alpha_nL in alpha_nl_array:
                for i in xrange(partitions):
                    if useruser is False:
                        suffix = '_similarity_ml100k_' + str(i) + '.mat'
                    else:
                        suffix = '_similarity_ml100k_' + str(i) + '_useruser' + '.mat'

                    helper.main_diffusion(matrixfunc, alpha_nL, 
                        threshold_fraction, transform=transform, 
                        simmat_suffix= suffix, 
                        output_prefix= str(i) + '_')

                dataFileName = "ml-100k/u.data"
                resultFileName = "results_" + vectorfunc + "vectorsim" + "_" \
                            + matrixfunc + "similaritymatrix_" + "alpha_nL_" + str(alpha_nL) + ".csv"
                args = " {0} {1} {2} {3}".format(dataFileName, resultFileName, vectorfunc, str(knn))
                subprocess.call(run_command + args, shell=True)


def run_test_set_partitions_diffusedutility(threshold_fraction=0.08, transform=helper.transformation_linear, 
                                            knn=40, numPartitions=5, reverse=False):
    for vectorfunc in vector_similarity_funcs:
        for matrixfunc in similarity_matrix_funcs:
            for alpha_nL in alpha_nl_array:
                diffuse_utility(numPartitions, alpha_nL, threshold_fraction, sim_func=matrixfunc, reverse=reverse)
                dataFileName = "ml-100k/u.data"
                resultFileName = "results_" + vectorfunc + "vectorsim" + "_" \
                            + matrixfunc + "similaritymatrix_" + "alpha_nL_" + str(alpha_nL) + ".csv"
                args = " {0} {1} {2} {3} {4}".format(dataFileName, resultFileName, vectorfunc, str(knn), 'double_diffusion')
                subprocess.call(run_command + args, shell=True)


# create similarity matrices in the current directory
def create_similarity_matrices(numUsers, numItems,source=filename,
    suffix='_similarity_ml100k.mat',file_delimiter='\t', useruser=False):
    for func in similarity_matrix_funcs:
        helper.main_similarity(func,source,numUsers,numItems,outsuffix=suffix, file_delimiter=file_delimiter,useruser=useruser)

def diffuse_utility(numPartitions, alpha_nL, threshold, sim_func,reverse):

    for i in xrange(numPartitions):
        helper.main_double_diffusion(input_file='ml-100k/u.data-crossfold/train.' + str(i) + '.csv', sim_func=sim_func,
                    alpha_nL=alpha_nL, threshold_fraction=threshold, transform=helper.transformation_linear, output_prefix=str(i) + '_',
                    reverse=reverse)

def analyse_csv(regular_algo_suffix='_itemitemCF'):
    # analyse the csv files
    printer = pprint.PrettyPrinter(indent=4)

    for vectorfunc in vector_similarity_funcs:
        for matrixfunc in similarity_matrix_funcs:

            aggregateFileName = vectorfunc + '_vectorsim_' + matrixfunc + 'similaritymatrix_aggregate.csv'
            aggregateFile = open(aggregateFileName, 'wb')
            writer = csv.writer(aggregateFile)
            first = True

            for alpha_nL in alpha_nl_array:
                resultFileName = "results_" + vectorfunc + "vectorsim" + "_" + matrixfunc + "similaritymatrix_" + "alpha_nL_" + str(alpha_nL) + ".csv"
                with open(resultFileName) as fi:
                    reader = csv.reader(fi)
                    result_matrix = []

                    for rownum, row in enumerate(reader):
                        if rownum == 0:
                            header = row
                        else:
                            result_matrix.append(row)
                    
                stats_by_diffusion_type = defaultdict(lambda: defaultdict(float))

                # collate stats from the results
                num_partitions = 0
                for row in result_matrix:
                    algorithmName = row[header.index('Algorithm')]
                    for metric in metrics:
                        #print row[header.index(metric)]
                        value = float(row[header.index(metric)])
                        stats_by_diffusion_type[algorithmName][metric] += value
                    stats_by_diffusion_type[algorithmName]['num_partitions'] += 1.0

                regular_algo_name = 'regular_' + vectorfunc + '_similarity' + regular_algo_suffix

                # average the stats
                for algorithmName, algorithmStats in stats_by_diffusion_type.iteritems():
                    for metric in metrics:
                        # print algorithmName, metric, algorithmStats[metric]
                        algorithmStats[metric] = algorithmStats[metric] / float(algorithmStats['num_partitions'])
                    assert(algorithmStats['num_partitions'] == 5.0)

                # print resultFileName
                # print "************ similarity_matrix: {0}  alpha_nL: {1} *************".format(matrixfunc, alpha_nL)
                # calculate the % error reduction

                # write the header
                if first is True:
                    print ['alpha_nL'] + list(chain.from_iterable(( algorithmName +'_'+ metric + '_improvement', algorithmName +'_'+ metric+'_average') \
                                    for algorithmName in sorted(stats_by_diffusion_type.keys()) for metric in metrics ))

                    writer.writerow(['alpha_nL'] + list(chain.from_iterable(( algorithmName +'_'+ metric + '_improvement', algorithmName +'_'+ metric+'_average') \
                                    for algorithmName in sorted(stats_by_diffusion_type.keys()) for metric in metrics )))
                    first = False

                row = [str(alpha_nL)]

                for algorithmName, algorithmStats in sorted(stats_by_diffusion_type.iteritems(), key = lambda x : x[0]):
                    # print algorithmName
                    for metric in metrics:
                        reg_error = stats_by_diffusion_type[regular_algo_name][metric]
                        error_reduction = (1.0 - algorithmStats[metric] / reg_error) * 100.0
                        algorithmStats[metric + '_error_reduction'] = error_reduction

                        row.append(str(error_reduction))
                        row.append(str(algorithmStats[metric]))


                writer.writerow(row)
                        #print metric + '_error_reduction', error_reduction
                        #print metric + '_average', algorithmStats[metric]


                    #print '#############################'


if __name__ == '__main__':

    # create_similarity_matrices(source='ml10M_reduced/ml1M_ml100k.dat',numUsers=6040, numItems=1682)

    # create_similarity_matrices(helper.defaultNumUsers,helper.defaultNumItems,'ml-100k/u.data')

    # helper.main_diffusion('cosine', 1.0, threshold_fraction=0.010)
    # subprocess.call("mv *.mat precomputed_matrices", shell=True)
    """
    for i in xrange(5):
        create_similarity_matrices(helper.defaultNumUsers, helper.defaultNumItems, 
            source='ml-100k/u.data-crossfold/train.' + str(i) + '.csv', 
            suffix='_similarity_ml100k_' + str(i) + '_useruser' + '.mat',
            file_delimiter=',', useruser=True)
    """
    """

    threshold = 0.08
    k = 50

    for transform, tname in [(helper.transformation_linear,'linear')]:
        for threshold in [0.08]:
            run_test_set_partitions(threshold_fraction=threshold, transform=transform, knn=k, useruser=True)
            analyse_csv()
            subprocess.call("mkdir results_threshold_" + str(threshold) + "k_" + str(k) + "_" + tname, shell=True)
            subprocess.call("mv *.csv results_threshold_" + str(threshold) + "k_" + str(k) + "_" + tname, shell=True)

    """

    run_test_set_partitions_diffusedutility(threshold_fraction=0.08, transform=helper.transformation_linear,
                                         knn=40, numPartitions=5, reverse=True)
    
    analyse_csv('_UserCF')

    # diffuse_utility(5, 1.0, 0.08, 'adjusted_cosine')


















import helper






class Test():

	def __init__(runcommand, datafile='ml-100k/u.data', similarity_matrix_funcs = ['adjusted_cosine'],
				vector_similarity_funcs = ['cosine'], alpha_nL_array=[0.1, 0.2, 0.4, 0.8, 1.0, 2.0, 4.0, 8.0, 100.0],
				partitions = 5, metrics = ['RMSE.ByUser','RMSE.ByRating','MAE.ByUser','MAE.ByRating']):
		self.runcommand = runcommand
		self.datafile = datafile
		self.similarity_matrix_funcs = similarity_matrix_funcs
		self.vector_similarity_funcs = vector_similarity_funcs
		self.alpha_nL_array = alpha_nL_array
		self.numPartitions = partitions
		self.metrics = metrics

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



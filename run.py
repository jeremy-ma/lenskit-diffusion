# some_file.py
import sys
import helper
import subprocess
import scipy.io
import numpy as np
import csv, pprint
from collections import defaultdict
from itertools import chain

run_command =  "/Library/Java/JavaVirtualMachines/jdk1.7.0_75.jdk/Contents/Home/bin/java -Didea.launcher.port=7537 \"-Didea.launcher.bin.path=/Applications/IntelliJ IDEA 14 CE.app/Contents/bin\" -Dfile.encoding=UTF-8 -classpath \"/Library/Java/JavaVirtualMachines/jdk1.7.0_75.jdk/Contents/Home/lib/ant-javafx.jar:/Library/Java/JavaVirtualMachines/jdk1.7.0_75.jdk/Contents/Home/lib/dt.jar:/Library/Java/JavaVirtualMachines/jdk1.7.0_75.jdk/Contents/Home/lib/javafx-doclet.jar:/Library/Java/JavaVirtualMachines/jdk1.7.0_75.jdk/Contents/Home/lib/javafx-mx.jar:/Library/Java/JavaVirtualMachines/jdk1.7.0_75.jdk/Contents/Home/lib/jconsole.jar:/Library/Java/JavaVirtualMachines/jdk1.7.0_75.jdk/Contents/Home/lib/sa-jdi.jar:/Library/Java/JavaVirtualMachines/jdk1.7.0_75.jdk/Contents/Home/lib/tools.jar:/Library/Java/JavaVirtualMachines/jdk1.7.0_75.jdk/Contents/Home/jre/lib/charsets.jar:/Library/Java/JavaVirtualMachines/jdk1.7.0_75.jdk/Contents/Home/jre/lib/deploy.jar:/Library/Java/JavaVirtualMachines/jdk1.7.0_75.jdk/Contents/Home/jre/lib/htmlconverter.jar:/Library/Java/JavaVirtualMachines/jdk1.7.0_75.jdk/Contents/Home/jre/lib/javaws.jar:/Library/Java/JavaVirtualMachines/jdk1.7.0_75.jdk/Contents/Home/jre/lib/jce.jar:/Library/Java/JavaVirtualMachines/jdk1.7.0_75.jdk/Contents/Home/jre/lib/jfr.jar:/Library/Java/JavaVirtualMachines/jdk1.7.0_75.jdk/Contents/Home/jre/lib/jfxrt.jar:/Library/Java/JavaVirtualMachines/jdk1.7.0_75.jdk/Contents/Home/jre/lib/jsse.jar:/Library/Java/JavaVirtualMachines/jdk1.7.0_75.jdk/Contents/Home/jre/lib/management-agent.jar:/Library/Java/JavaVirtualMachines/jdk1.7.0_75.jdk/Contents/Home/jre/lib/plugin.jar:/Library/Java/JavaVirtualMachines/jdk1.7.0_75.jdk/Contents/Home/jre/lib/resources.jar:/Library/Java/JavaVirtualMachines/jdk1.7.0_75.jdk/Contents/Home/jre/lib/rt.jar:/Library/Java/JavaVirtualMachines/jdk1.7.0_75.jdk/Contents/Home/jre/lib/ext/dnsns.jar:/Library/Java/JavaVirtualMachines/jdk1.7.0_75.jdk/Contents/Home/jre/lib/ext/localedata.jar:/Library/Java/JavaVirtualMachines/jdk1.7.0_75.jdk/Contents/Home/jre/lib/ext/sunec.jar:/Library/Java/JavaVirtualMachines/jdk1.7.0_75.jdk/Contents/Home/jre/lib/ext/sunjce_provider.jar:/Library/Java/JavaVirtualMachines/jdk1.7.0_75.jdk/Contents/Home/jre/lib/ext/sunpkcs11.jar:/Library/Java/JavaVirtualMachines/jdk1.7.0_75.jdk/Contents/Home/jre/lib/ext/zipfs.jar:/Users/jeremyma/Documents/Research/lenskit-algorithm-example-master/build/classes/main:/Users/jeremyma/Documents/Research/lenskit-algorithm-example-master/build/resources/main:/Users/jeremyma/.gradle/caches/modules-2/files-2.1/org.grouplens.lenskit/lenskit-all/2.2-M3/2d905cb1afd4ebb6da82bd506d5cc998da952797/lenskit-all-2.2-M3.jar:/Users/jeremyma/.gradle/caches/modules-2/files-2.1/org.grouplens.lenskit/lenskit-knn/2.2-M3/554c8537e4f71685359285bbec857e8db18e61ad/lenskit-knn-2.2-M3.jar:/Users/jeremyma/.gradle/caches/modules-2/files-2.1/org.grouplens.lenskit/lenskit-svd/2.2-M3/8ca52a6b54121b92469c7095a9c224cc7af401a7/lenskit-svd-2.2-M3.jar:/Users/jeremyma/.gradle/caches/modules-2/files-2.1/org.grouplens.lenskit/lenskit-slopeone/2.2-M3/ad7eb8f58d5dd33aeaa0b3d5c3b0b54c2e03b83/lenskit-slopeone-2.2-M3.jar:/Users/jeremyma/.gradle/caches/modules-2/files-2.1/org.grouplens.lenskit/lenskit-eval/2.2-M3/23518e395de3bb703a3e214f6c205f958876588e/lenskit-eval-2.2-M3.jar:/Users/jeremyma/.gradle/caches/modules-2/files-2.1/org.grouplens.lenskit/lenskit-predict/2.2-M3/3b351092c4fc9baf4aae72f10f8a2120eceeafbb/lenskit-predict-2.2-M3.jar:/Users/jeremyma/.gradle/caches/modules-2/files-2.1/org.grouplens.lenskit/lenskit-core/2.2-M3/bc5852ebd02025cc8b7ac7f3202fdba21a78fe7d/lenskit-core-2.2-M3.jar:/Users/jeremyma/.gradle/caches/modules-2/files-2.1/org.slf4j/slf4j-api/1.7.6/562424e36df3d2327e8e9301a76027fca17d54ea/slf4j-api-1.7.6.jar:/Users/jeremyma/.gradle/caches/modules-2/files-2.1/com.google.code.findbugs/annotations/2.0.1/9ef6656259841cebfb9fb0697bb122ada4485498/annotations-2.0.1.jar:/Users/jeremyma/.gradle/caches/modules-2/files-2.1/joda-time/joda-time/2.3/56498efd17752898cfcc3868c1b6211a07b12b8f/joda-time-2.3.jar:/Users/jeremyma/.gradle/caches/modules-2/files-2.1/org.apache.ant/ant/1.8.4/8acff3fb57e74bc062d4675d9dcfaffa0d524972/ant-1.8.4.jar:/Users/jeremyma/.gradle/caches/modules-2/files-2.1/commons-cli/commons-cli/1.2/2bf96b7aa8b611c177d329452af1dc933e14501c/commons-cli-1.2.jar:/Users/jeremyma/.gradle/caches/modules-2/files-2.1/org.grouplens.lenskit/lenskit-groovy/2.2-M3/4a17bee262e14617e286bcbeab1504d3d82ace37/lenskit-groovy-2.2-M3.jar:/Users/jeremyma/.gradle/caches/modules-2/files-2.1/org.codehaus.groovy/groovy-all/2.1.5/eda9522cc90f16c06dd51739e2d02daafad0b36f/groovy-all-2.1.5.jar:/Users/jeremyma/.gradle/caches/modules-2/files-2.1/org.hamcrest/hamcrest-library/1.3/4785a3c21320980282f9f33d0d1264a69040538f/hamcrest-library-1.3.jar:/Users/jeremyma/.gradle/caches/modules-2/files-2.1/net.mikera/vectorz/0.41.2/88910ca2ff34134b23836c3e28f6970e3cb537c7/vectorz-0.41.2.jar:/Users/jeremyma/.gradle/caches/modules-2/files-2.1/com.google.guava/guava/18.0/cce0823396aa693798f8882e64213b1772032b09/guava-18.0.jar:/Users/jeremyma/.gradle/caches/modules-2/files-2.1/org.apache.commons/commons-compress/1.8/5bcbb3368441e4ced9570e24c006cd207a478bac/commons-compress-1.8.jar:/Users/jeremyma/.gradle/caches/modules-2/files-2.1/org.grouplens.lenskit/lenskit-api/2.2-M3/130e463179d150ee65733894c896def00d683bb0/lenskit-api-2.2-M3.jar:/Users/jeremyma/.gradle/caches/modules-2/files-2.1/org.grouplens.grapht/grapht/0.10.0-BETA1/cd964723f2f148484a949c5521078313cad41394/grapht-0.10.0-BETA1.jar:/Users/jeremyma/.gradle/caches/modules-2/files-2.1/org.apache.ant/ant-launcher/1.8.4/22f1e0c32a2bfc8edd45520db176bac98cebbbfe/ant-launcher-1.8.4.jar:/Users/jeremyma/.gradle/caches/modules-2/files-2.1/org.hamcrest/hamcrest-core/1.3/42a25dc3219429f0e5d060061f71acb49bf010a0/hamcrest-core-1.3.jar:/Users/jeremyma/.gradle/caches/modules-2/files-2.1/us.bpsm/edn-java/0.4.4/3a076e4dab444a4f12350c55690a86c3f5684cc3/edn-java-0.4.4.jar:/Users/jeremyma/.gradle/caches/modules-2/files-2.1/net.mikera/randomz/0.3.0/c7f52025be1165cf29752263f41dc173d4489cc5/randomz-0.3.0.jar:/Users/jeremyma/.gradle/caches/modules-2/files-2.1/net.mikera/mathz/0.3.0/80a1c38c038343721104d793c72905ef6cf6226d/mathz-0.3.0.jar:/Users/jeremyma/.gradle/caches/modules-2/files-2.1/org.tukaani/xz/1.5/9c64274b7dbb65288237216e3fae7877fd3f2bee/xz-1.5.jar:/Users/jeremyma/.gradle/caches/modules-2/files-2.1/org.grouplens.lenskit/lenskit-data-structures/2.2-M3/be05df5277e15b8d3fbe4988bf837fcc65f124ee/lenskit-data-structures-2.2-M3.jar:/Users/jeremyma/.gradle/caches/modules-2/files-2.1/javax.inject/javax.inject/1/6975da39a7040257bd51d21a231b76c915872d38/javax.inject-1.jar:/Users/jeremyma/.gradle/caches/modules-2/files-2.1/com.google.code.findbugs/jsr305/1.3.9/40719ea6961c0cb6afaeb6a921eaa1f6afd4cfdf/jsr305-1.3.9.jar:/Users/jeremyma/.gradle/caches/modules-2/files-2.1/it.unimi.dsi/fastutil/6.6.1/92c455773efaa4f55d48f70a8d24d7fde9d67519/fastutil-6.6.1.jar:/Users/jeremyma/.gradle/caches/modules-2/files-2.1/org.apache.commons/commons-lang3/3.3/5ccde9cb5e3071eaadf5d87a84b4d0aba43b119/commons-lang3-3.3.jar:/Users/jeremyma/.gradle/caches/modules-2/files-2.1/ch.qos.logback/logback-classic/1.1.2/b316e9737eea25e9ddd6d88eaeee76878045c6b2/logback-classic-1.1.2.jar:/Users/jeremyma/.gradle/caches/modules-2/files-2.1/ch.qos.logback/logback-core/1.1.2/2d23694879c2c12f125dac5076bdfd5d771cc4cb/logback-core-1.1.2.jar:/Users/jeremyma/Documents/Research/JMatIO-041212/lib/jmatio.jar:/Users/jeremyma/Documents/Research/commons-math3-3.5/commons-math3-3.5.jar:/Applications/IntelliJ IDEA 14 CE.app/Contents/lib/idea_rt.jar\" com.intellij.rt.execution.application.AppMain org.grouplens.lenskit.hello.HelloLenskit"

# run_command = "java -jar ./build/classes/artifacts/lenskit_algorithm_example_master_jar2/lenskit-algorithm-example-master.jar"

filename = 'ml-100k/u.data'
similarity_matrix_funcs = ['cosine', 'adjusted_cosine', 'pearson']
vector_similarity_funcs = ['cosine']
alpha_nl_array = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 100.0, 1000.0, 10e9]
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
def run_test_set_partitions(threshold_fraction=0.3, transform=helper.transformation_linear):
	for vectorfunc in vector_similarity_funcs:
		for matrixfunc in similarity_matrix_funcs:
			for alpha_nL in alpha_nl_array:
				for i in xrange(partitions):
					helper.main_diffusion(matrixfunc, alpha_nL, 
						threshold_fraction, transform, 
						simmat_suffix='_similarity_ml100k_' + str(i) + '.mat', 
						output_prefix= str(i) + '_')

				dataFileName = "ml-100k/u.data"
				resultFileName = "results_" + vectorfunc + "vectorsim" + "_" \
							+ matrixfunc + "similaritymatrix_" + "alpha_nL_" + str(alpha_nL) + ".csv"
				args = " {0} {1} {2}".format(dataFileName, resultFileName, vectorfunc)
				subprocess.call(run_command + args, shell=True)


# create similarity matrices in the current directory
def create_similarity_matrices(numUsers, numItems,source=filename,
	suffix='_similarity_ml100k.mat',file_delimiter='\t'):
	for func in similarity_matrix_funcs:
		helper.main_similarity(func,source,numUsers,numItems,outsuffix=suffix, file_delimiter=file_delimiter)


def analyse_csv():
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

				regular_algo_name = 'regular_' + vectorfunc + '_similarity'

				# average the stats
				for algorithmName, algorithmStats in stats_by_diffusion_type.iteritems():
					for metric in metrics:
						# print algorithmName, metric, algorithmStats[metric]
						algorithmStats[metric] = algorithmStats[metric] / float(algorithmStats['num_partitions'])
					assert(algorithmStats['num_partitions'] == 5.0)


				#print resultFileName
				#print "************ similarity_matrix: {0}  alpha_nL: {1} *************".format(matrixfunc, alpha_nL)
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

	"""
	for i in xrange(5):
		create_similarity_matrices(helper.defaultNumUsers, helper.defaultNumItems, 
			source='ml-100k/u.data-crossfold/train.' + str(i) + '.csv', 
			suffix='_similarity_ml100k_' + str(i) + '.mat',
			file_delimiter=',')
	"""

	run_test_set_partitions(threshold_fraction=0.05, transform=helper.transformation_squared)


	# run_test_set(threshold_fraction=0.15, transform=helper.transformation_cubed)
	# analyse_csv()

	# subprocess.call("mkdir results_threshold_0.15_cubed", shell=True)
	# subprocess.call("mv *.csv results_threshold_0.15_cubed", shell=True)

















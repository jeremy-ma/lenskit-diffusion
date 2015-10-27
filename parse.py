# Parse results in nested directory format
import os,re,pdb,sys
from collections import defaultdict
import csv

metrics = ['RMSE.ByUser','RMSE.ByRating','MAE.ByUser','MAE.ByRating']

regex = r'knn_(\d+)_alphanL_([\d\.]+)_threshold_([\d\.]+)\.csv'
reggie = re.compile(regex)

def read_csv(filename):
	#returns results in dict

	results = dict(zip(metrics,[0.0 for i in xrange(len(metrics))]))
	num_partitions = 0.0

	with open(filename) as fi:

		for ind,line in enumerate(fi):
			if ind == 0:
				# get a mapping of column metric to its column index
				headers = line.rstrip().split(',')
				indices = [i for i,_ in enumerate(headers)]
				metric_to_col= dict(zip(headers,indices))
			else:
				#aggregate the results for each partition
				num_partitions += 1.0
				# separate the line
				separated = line.rstrip().split(',')
				# add the results for each metric
				for metric in metrics:
					results[metric] += float(separated[metric_to_col[metric]])

	# calculate the average
	for metric, val in results.iteritems():
		results[metric] = val / num_partitions

	return results


def summarise(root, files):
	results = defaultdict(dict)

	for fname in files:
		if fname == '.DS_Store':
			continue
		match = reggie.match(fname)
		if match is None:
			continue
		result = read_csv(root+'/'+fname)
		knn = int(match.group(1))
		alpha = float(match.group(2))
		threshold = float(match.group(3))
		#store by knn and threshold setting
		results[(knn,threshold)][alpha] = result


	for (knn,threshold), val in results.iteritems():
		alpha_results = [(k, v) for k, v in val.iteritems()]
		alpha_results.sort()
		aggregatename = "aggregate_knn_{0}_threshold_{1}.csv".format(knn,threshold)
		with open(root + '/' + aggregatename,'w') as fi:
			writer = csv.writer(fi)
			header = ['alpha'] + metrics
			writer.writerow(header)
			#pdb.set_trace()
			for alpha, res in alpha_results:
				#print res
				errors = [str(res[metric]) for metric in metrics]
				row = [str(alpha)] + errors
				writer.writerow(row)



def parse(startdir):
	# traverse root directory, and list directories as dirs and files as files
	for root, dirs, files in os.walk(startdir):
	    path = root.split('/')

	    if len(files) > 1:
		    summarise(root,files)


if __name__ == '__main__':
	name = sys.argv[1]
	parse(name)




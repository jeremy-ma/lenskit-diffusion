# some_file.py
import sys
sys.path.insert(0, '/Users/jeremyma/Documents/Research/preprocessing/usersimilarity')
import similarity_matrices


if __name__ == '__main__':

	# create matrix

	alpha_nl_array = [0.5, 1.0, 2.0, 4.0, 10.0, 100.0, 1000.0, 10e9]


	for alpha_nl in alpha_nl_array:
		# create similarity matrices
		similarity_matrices.main(alpha_nl)




















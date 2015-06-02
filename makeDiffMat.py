import numpy as np
import scipy.io
from scipy.sparse import csgraph


if __name__ == '__main__':
    diffmat = scipy.io.loadmat('ml100k_difference.mat')['ml100k_difference']

    #set zeros to 0.001
    diffmat[diffmat==0] = 0.001
    simmat = 1/diffmat
    simmat[np.diag_indices(len(simmat))] = 0

    L = csgraph.laplacian(simmat, normed=False)
    L_n = csgraph.laplacian(simmat, normed=True)

    # calculate diffusion rates
    #set alpha_nL
    alpha_nL = 0.01

    ratio_diagL_diagNL = L.diagonal().sum() / L_n.diagonal().sum()
    alpha_L = alpha_nL / ratio_diagL_diagNL


    diff = np.linalg.inv(np.eye(len(simmat)) + alpha_L * L)
    diff_n = np.linalg.inv(np.eye(len(simmat)) + alpha_nL * L_n)

    scipy.io.savemat('ml100k_diff.mat',mdict={'ml100k_diff':diff})
    scipy.io.savemat('ml100k_diff_n.mat',mdict={'ml100k_diff_n':diff_n})

    #b = scipy.io.loadmat('ml100k_diff.mat')







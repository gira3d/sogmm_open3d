import sogmm_open3d_py
import numpy as np
from termcolor import colored

def helper(K, DD):
    D = int(np.sqrt(DD))
    P = np.zeros((K, DD))

    count = 0
    for i in range(0, K):
        for j in range(0,DD):
            P[i,j] = count
            count += 1

    return P

###############################
#          Test 1
###############################
A = np.loadtxt('./test_data/covs.txt')
crt_answer = np.loadtxt('./test_data/compute_cholesky_python_output.txt')
answer = sogmm_open3d_py.compute_cholesky_for_loop(A)
np.testing.assert_array_almost_equal(crt_answer, answer, decimal=6)
print(colored('Test 1 Passed', 'green'))

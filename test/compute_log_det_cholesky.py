import sogmm_open3d_py
import numpy as np
from termcolor import colored

###############################
#          Test 1
###############################
P = np.loadtxt('./test_data/precisions_cholesky.csv', delimiter=",")
answer = sogmm_open3d_py.compute_log_det_cholesky(P)
crt_answer = np.loadtxt('./test_data/python_output.csv', delimiter=",")
np.testing.assert_array_almost_equal(crt_answer, answer.flatten(), decimal=6)
print(colored('Test 1 Passed', 'green'))

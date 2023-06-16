import sogmm_open3d_py
import numpy as np
from termcolor import colored

###############################
#          Test 1
###############################
P = np.loadtxt('./test_data/score_samples_test__precs_chol.txt')
X = np.loadtxt('./test_data/score_samples_test__X.txt')
M = np.loadtxt('./test_data/score_samples_test__means.txt')
W = np.loadtxt('./test_data/score_samples_test__weights.txt')

answer = sogmm_open3d_py.score_samples(X, M, W, P)
crt_answer = np.loadtxt('./test_data/score_samples_test__python_output.txt', delimiter=",")
np.testing.assert_array_almost_equal(crt_answer, answer.flatten(), decimal=5)
print(colored('Test 1 Passed', 'green'))

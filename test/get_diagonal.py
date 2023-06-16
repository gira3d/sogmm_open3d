import sogmm_open3d_py
import numpy as np
from termcolor import colored


def run_test(K, DD):
    D = int(np.sqrt(DD))
    P = np.zeros((K, DD))

    count = 0
    for i in range(0, K):
        for j in range(0, DD):
            P[i, j] = count
            count += 1
    P_tensor = P.reshape((K, D, D))
    answer = sogmm_open3d_py.diagonal(P)

    crt_answer = np.zeros((np.shape(P_tensor)[0], np.shape(P_tensor)[1]))
    for i in range(0, np.shape(P_tensor)[0]):
        crt_answer[i, :] = np.diag(P_tensor[i, :, :])

    np.testing.assert_array_almost_equal(crt_answer, answer, decimal=8)


###############################
#          Test 1
###############################
run_test(5, 16)
print(colored('Test 1 Passed', 'green'))

###############################
#          Test 2
###############################
run_test(7, 16)
print(colored('Test 2 Passed', 'green'))

###############################
#          Test 3
###############################
run_test(18, 9)
print(colored('Test 3 Passed', 'green'))

###############################
#          Test 4
###############################
run_test(6, 4)
print(colored('Test 4 Passed', 'green'))

###############################
#          Test 5
###############################
run_test(3, 16)
print(colored('Test 5 Passed', 'green'))

import sogmm_open3d_py
import numpy as np
from termcolor import colored

def createA(n):
    A = np.zeros((n,n))
    count = 0
    for i in range(0,n):
        for j in range(0,n):
            A[i,j] = count
            count = count+1
    return A

###############################
#          Test 1
###############################
A = createA(5)
crt_answer = [ 4.4519143,  9.451915,  14.451915,  19.451914,  24.451914 ]

answer = sogmm_open3d_py.log_sum_exp_cols(A).transpose()[0]
np.testing.assert_array_almost_equal(crt_answer, answer, decimal=6)
print(colored('Test 1 Passed', 'green'))

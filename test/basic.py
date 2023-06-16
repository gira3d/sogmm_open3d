import sogmm_open3d_py
import numpy as np

A = np.zeros((3,3))
A[0,0] = 4
A[0,1] = 12
A[0,2] = -16

A[1,0] = 12
A[1,1] = 37
A[1,2] = -43

A[2,0] = -16
A[2,1] = -43
A[2,2] = 98

sogmm_open3d_py.test(A)
print('test complete')

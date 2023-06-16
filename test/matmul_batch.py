import time
import argparse
import numpy as np
from termcolor import cprint

import sogmm_open3d_py as sop

class MatmulBatchTest:
    def __init__(self, N, K, D):
        self.K = K
        self.D = D
        self.C = self.D * self.D

    def test_3d(self):
        A = np.random.rand(self.K, self.D, self.D)
        B = np.random.rand(self.K, self.D, self.D)

        crt_ans = np.zeros(A.shape)
        start = time.time()
        for i in range(self.K):
            crt_ans[i, :, :] = np.matmul(A[i, :, :], B[i, :, :])
        end = time.time()
        cprint('numpy matmul 3d time %f seconds' % (end - start), 'yellow')

        output = sop.mat_mul_batched(A.reshape((self.K, self.C)), B.reshape((self.K, self.C)))
        np.testing.assert_array_almost_equal(crt_ans, output.reshape(crt_ans.shape))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Matmul Batched Test")
    parser.add_argument('--samples', type=int, default=10)
    parser.add_argument('--components', type=int, default=2)
    parser.add_argument('--dim', type=int, default=2)
    args = parser.parse_args()

    mb = MatmulBatchTest(args.samples, args.components, args.dim)

    mb.test_3d()
import numpy as np
from cprint import cprint
import unittest
from config_parser import ConfigParser
import time

from sogmm_py.utils import matrix_to_tensor, o3d_to_np, calculate_depth_metrics, np_to_o3d, tensor_to_matrix
from sogmm_py.utils import read_log_trajectory, ImageUtils
from sogmm_py.vis_open3d import VisOpen3D

# Datasets
from sklearn import datasets

# Plotting/Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt

# Point cloud fit
import os
import open3d as o3d

import sogmm_cpu
from sogmm_cpu import SOGMMf4Host as CPUContainerf4
from sogmm_cpu import SOGMMf3Host as CPUContainerf3
from sogmm_cpu import SOGMMf2Host as CPUContainerf2
from sogmm_cpu import SOGMMLearner as CPUFit
from sogmm_cpu import SOGMMInference as CPUInference

import sogmm_gpu
from sogmm_gpu import SOGMMf4Device as GPUContainerf4
from sogmm_gpu import SOGMMf3Device as GPUContainerf3
from sogmm_gpu import SOGMMf2Device as GPUContainerf2
from sogmm_gpu import SOGMMLearner as GPUFit
from sogmm_gpu import SOGMMInference as GPUInference

# GMR Python
from gmr import GMM, plot_error_ellipses


class GPUImpl:
    '''GPU implementation'''

    def __init__(self, b):
        self.learner = GPUFit(b)
        self.inference = GPUInference()


class SOGMMCPUTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # configs
        parser = ConfigParser()
        cls.config = parser.get_config()
        cls.plot_show = cls.config.show_plots

        # datasets
        cls.data_2d_1, _ = datasets.make_blobs(
            n_samples=cls.config.n_samples, centers=cls.config.n_components, random_state=10)
        cls.data_2d_2, _ = datasets.make_blobs(
            n_samples=cls.config.n_samples, centers=cls.config.n_components, random_state=20)

        # poses
        traj_file = os.path.join(
            cls.config.path_dataset, f"{cls.config.dataset}-traj.log")
        cls.traj = read_log_trajectory(traj_file)

        # utility
        K = np.eye(3)
        K[0, 0] = 525.0
        K[1, 1] = 525.0
        K[0, 2] = 319.5
        K[1, 2] = 239.5

        cls.iu = ImageUtils(K)

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        # start runners at the beginning of every test
        self.cpp = GPUImpl(self.config.bandwidth)

    def tearDown(self):
        # deleter runners at the beginning of every test
        del self.cpp

    @unittest.skip("Skipping test_merging")
    def test_merging(self):
        gmm1 = GMM(n_components=self.config.n_components)
        gmm1.from_samples(self.data_2d_1)
        gmm2 = GMM(n_components=self.config.n_components)
        gmm2.from_samples(self.data_2d_2)

        sogmm_cpu1 = CPUContainerf2(gmm1.priors, gmm1.means, tensor_to_matrix(
            gmm1.covariances, 2), self.data_2d_1.shape[0])
        cprint.ok(
            f"loaded sogmm_cpu1 with {sogmm_cpu1.n_components_} components.")
        sogmm_cpu2 = CPUContainerf2(gmm2.priors, gmm2.means, tensor_to_matrix(
            gmm2.covariances, 2), self.data_2d_2.shape[0])
        cprint.ok(
            f"loaded sogmm_cpu2 with {sogmm_cpu2.n_components_} components.")

        sogmm_gpu1 = GPUContainerf2()
        sogmm_gpu1.from_host(sogmm_cpu1)
        sogmm_gpu2 = GPUContainerf2()
        sogmm_gpu2.from_host(sogmm_cpu2)
        cprint.ok(f"loaded both on the GPU.")

        # merge on CPU
        sogmm_cpu2.merge(sogmm_cpu1)
        cprint.ok(f"merged on CPU.")

        # merge on GPU
        sogmm_gpu2.merge(sogmm_gpu1)
        cprint.ok(f"merged on GPU.")

        sogmm_cpu2_conv = CPUContainerf2()
        sogmm_gpu2.to_host(sogmm_cpu2_conv)

        # test if the merging operation is identical on CPU and GPU
        np.testing.assert_almost_equal(
            sogmm_cpu2.weights_, sogmm_cpu2_conv.weights_)
        np.testing.assert_almost_equal(
            sogmm_cpu2.means_, sogmm_cpu2_conv.means_)
        np.testing.assert_almost_equal(
            sogmm_cpu2.covariances_, sogmm_cpu2_conv.covariances_)
        np.testing.assert_almost_equal(
            sogmm_cpu2.precisions_cholesky_, sogmm_cpu2_conv.precisions_cholesky_)
        np.testing.assert_almost_equal(
            sogmm_cpu2.covariances_cholesky_, sogmm_cpu2_conv.covariances_cholesky_)
        cprint.ok(f"Tested merging.")

    @unittest.skip("Skipping test_marginal_gpu")
    def test_marginal_gpu(self):
        pcld_o3d = o3d.io.read_point_cloud(os.path.join(self.config.path_dataset,
                                                        f"pcd_{self.config.dataset}_{0}_decimate_2_0.pcd"),
                                           format='pcd')
        pcld = o3d_to_np(pcld_o3d)

        d = np.array([np.linalg.norm(x)
                      for x in pcld[:, 0:3]])[:, np.newaxis]
        g = pcld[:, 3][:, np.newaxis]
        Y = np.concatenate((d, g), axis=1)

        sogmm = GPUContainerf4()
        self.cpp.learner.fit(Y, pcld, sogmm)
        cprint.ok(
            f"GPU Learner number of components: {sogmm.n_components_}")
        sogmm4_cpu = CPUContainerf4(sogmm.n_components_)
        sogmm.to_host(sogmm4_cpu)
        ref = sogmm_cpu.marginal_X(sogmm4_cpu)

        sogmm3 = sogmm_gpu.marginal_X(sogmm)
        sogmm3_cpu = CPUContainerf3(sogmm3.n_components_)
        sogmm3.to_host(sogmm3_cpu)

        np.testing.assert_almost_equal(sogmm3_cpu.weights_, ref.weights_)
        np.testing.assert_almost_equal(sogmm3_cpu.means_, ref.means_)
        np.testing.assert_almost_equal(sogmm3_cpu.covariances_, ref.covariances_)


    def test_likelihood_scores_gpu(self):
        pcld_o3d_1 = o3d.io.read_point_cloud(os.path.join(self.config.path_dataset,
                                                        f"pcd_{self.config.dataset}_{0}_decimate_2_0.pcd"),
                                           format='pcd')
        pcld_1 = o3d_to_np(pcld_o3d_1)
        pcld_o3d_2 = o3d.io.read_point_cloud(os.path.join(self.config.path_dataset,
                                                        f"pcd_{self.config.dataset}_{50}_decimate_2_0.pcd"),
                                           format='pcd')
        pcld_2 = o3d_to_np(pcld_o3d_2)

        d = np.array([np.linalg.norm(x)
                      for x in pcld_1[:, 0:3]])[:, np.newaxis]
        g = pcld_1[:, 3][:, np.newaxis]
        Y = np.concatenate((d, g), axis=1)

        sogmm = GPUContainerf4()
        self.cpp.learner.fit(Y, pcld_1, sogmm)
        cprint.ok(
            f"GPU Learner number of components: {sogmm.n_components_}")

        scores = self.cpp.inference.score_3d(pcld_2[:, :3], sogmm)
        scores = scores.flatten()
        novel_pts = pcld_2[scores < -10.0, :3]

        vis = VisOpen3D(visible=True)
        vis.add_geometry(pcld_o3d_1)
        vis.add_geometry(np_to_o3d(novel_pts, color=[1.0, 0.0, 0.0]))
        vis.render()


    @unittest.skip("Skipping test_4d_conditional")
    def test_4d_conditional(self):
        vis = VisOpen3D(visible=True)
        for i in range(0, 1):
            cprint.info(f"processing frame {i}")
            pcld_o3d = o3d.io.read_point_cloud(os.path.join(self.config.path_dataset,
                                                            f"pcd_{self.config.dataset}_{i}_decimate_2_0.pcd"),
                                               format='pcd')
            pcld = o3d_to_np(pcld_o3d)

            d = np.array([np.linalg.norm(x)
                         for x in pcld[:, 0:3]])[:, np.newaxis]
            g = pcld[:, 3][:, np.newaxis]
            Y = np.concatenate((d, g), axis=1)

            sogmm = GPUContainerf4()
            self.cpp.learner.fit(Y, pcld, sogmm)
            cprint.ok(
                f"GPU Learner number of components: {sogmm.n_components_}")
            sogmm_cpu = CPUContainerf4()
            sogmm.to_host(sogmm_cpu)
            cprint.info(f"Copied to host: {sogmm.n_components_}")

            ts = time.time()
            points4d = self.cpp.inference.reconstruct(sogmm,
                                                      4 * sogmm.support_size_,
                                                      2.2)
            te = time.time()
            cprint.info(
                f"[cpp cpu] time taken for 3D points and inference: {te - ts}")
            vis.add_geometry(np_to_o3d(points4d))

            pose = self.traj[i].pose
            vis.update_view_point(extrinsic=np.linalg.inv(pose))

        vis.render()


if __name__ == "__main__":
    unittest.main()

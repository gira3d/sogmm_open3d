import numpy as np
import pickle
import open3d as o3d
import time
import matplotlib as mpl
import matplotlib.cm as cm

from sogmm_py.utils import o3d_to_np, np_to_o3d
from sogmm_py.sogmm import SOGMM
from sogmm_py.vis_open3d import VisOpen3D

cmap = mpl.colormaps['viridis']

sg = SOGMM(0.02, compute='GPU')
pose = np.loadtxt('test_color_conditional_pose.txt')

with open('model.pkl', 'rb') as f:
    sg.model = pickle.load(f)

pcd = o3d.io.read_point_cloud('test_color_conditional.pcd', format='pcd')
np_pcd = o3d_to_np(pcd)

# st = time.time()
# py_output_ws, py_output_ms, py_output_covs = sg.color_conditional(np_pcd[:, 0:3])
# en = time.time()
# print('python time %f' % (en - st))

st = time.time()
cpp_output_ws, cpp_output_ms, cpp_output_covs = sg.model.color_conditional(np_pcd[:, 0:3])
en = time.time()
print('cpp time %f' % (en - st))

# if py_output_ws.ndim == 1:
#     py_output_ws = py_output_ws[:, np.newaxis]

# if cpp_output_ws.ndim == 1:
#     cpp_output_ws = cpp_output_ws[:, np.newaxis]

# assert py_output_ws.shape == cpp_output_ws.shape, f'cpp output shape (%s) does not match python output shape (%s)' % (
#     cpp_output_ws.shape, py_output_ws.shape)

# if py_output_ms.ndim == 1:
#     py_output_ms = py_output_ms[:, np.newaxis]

# if cpp_output_ms.ndim == 1:
#     cpp_output_ms = cpp_output_ms[:, np.newaxis]

# assert py_output_ms.shape == cpp_output_ms.shape, f'cpp output shape (%s) does not match python output shape (%s)' % (
#     cpp_output_ms.shape, py_output_ms.shape)

# if py_output_covs.ndim == 1:
#     py_output_covs = py_output_covs[:, np.newaxis]

# if cpp_output_covs.ndim == 1:
#     cpp_output_covs = cpp_output_covs[:, np.newaxis]

# assert py_output_covs.shape == cpp_output_covs.shape, f'cpp output shape (%s) does not match python output shape (%s)' % (
#     cpp_output_covs.shape, py_output_covs.shape)

# print('asserting weights python and cpp match')
# np.testing.assert_array_almost_equal(py_output_ws, cpp_output_ws, decimal=3)

# print('asserting means python and cpp match')
# np.testing.assert_array_almost_equal(py_output_ms, cpp_output_ms, decimal=4)

# print('asserting covs python and cpp match')
# np.testing.assert_array_almost_equal(py_output_covs, cpp_output_covs, decimal=4)

print('number of components are', sg.model.n_components_)

recon_pcd = np.zeros(np_pcd.shape)
recon_pcd[:, :3] = np_pcd[:, :3]
recon_pcd[:, 3] = np.squeeze(cpp_output_ms)

uncert_pcd = np.zeros((np_pcd.shape[0], 6))
uncert_pcd[:, :3] = np_pcd[:, :3]
uncerts = np.squeeze(cpp_output_covs)

minima = np.min(uncerts)
maxima = np.max(uncerts)

norm = mpl.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
mapper = cm.ScalarMappable(norm=norm, cmap=cmap)

uncert_pcd[:, 3:6] = np.array(mapper.to_rgba(uncerts))[:, 0:3]

# visualize reconstruction
O3D_K = np.array([[935.30743609,   0.,         959.5],
                    [0.,         935.30743609, 539.5],
                    [0.,           0.,           1.]])
vis = VisOpen3D(visible=True)
vis.add_geometry(np_to_o3d(recon_pcd))
vis.update_view_point(O3D_K, np.linalg.inv(pose))
vis.update_renderer()
vis.run()
del vis

# visualize uncertainty image
vis = VisOpen3D(visible=True)
vis.add_geometry(np_to_o3d(uncert_pcd))
vis.update_view_point(O3D_K, np.linalg.inv(pose))
vis.update_renderer()
vis.run()
del vis
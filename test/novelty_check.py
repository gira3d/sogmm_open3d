import os
import time
import pathlib
import numpy as np
from termcolor import colored
import glob
import open3d as o3d
import argparse
from termcolor import cprint
from sklearn.mixture import GaussianMixture as PythonGMM

import sogmm_open3d_py
from sogmm_py.utils import ImageUtils, read_log_trajectory, np_to_o3d
from sogmm_py.vis_open3d import VisOpen3D

def generate_red_pcld(pcld_in):
    if np.shape(pcld_in)[0] == 0:
        cprint('Returning empty pointcloud', 'yellow')
        return o3d.geometry.PointCloud()

    pcld_out = o3d.geometry.PointCloud()
    pcld_out.points = o3d.utility.Vector3dVector(pcld_in)
    pcld_out.paint_uniform_color([1, 0.0, 0.0])
    return pcld_out

def generate_blue_pcld(pcld_in):
    if np.shape(pcld_in)[0] == 0:
        cprint('Returning empty pointcloud', 'yellow')
        return o3d.geometry.PointCloud()

    pcld_out = o3d.geometry.PointCloud()
    pcld_out.points = o3d.utility.Vector3dVector(pcld_in)
    pcld_out.paint_uniform_color([0.0, 0.0, 1.0])
    return pcld_out

def generate_green_pcld(pcld_in):
    if np.shape(pcld_in)[0] == 0:
        cprint('Returning empty pointcloud', 'yellow')
        return o3d.geometry.PointCloud()

    pcld_out = o3d.geometry.PointCloud()
    pcld_out.points = o3d.utility.Vector3dVector(pcld_in)
    pcld_out.paint_uniform_color([0.0, 1.0, 0.0])
    return pcld_out

def get3DPart(gmm_in):
    means = gmm_in['means'][:,0:3]
    weights = gmm_in['weights']
    precs_chol_4d = np.reshape(gmm_in['precs_chol'], (len(weights), 4, 4))
    precs_chol = np.zeros((len(weights), 3, 3))
    for i in range(0, len(weights)):
        precs_chol[i,:,:] = precs_chol_4d[i, 0:3, 0:3]
    precs_chol = np.reshape(precs_chol, (len(weights), 9))
    return (means, weights, precs_chol)

def novelty_check(pin, scored_samples, novelty_thresh):

    count = 0
    for i in range(0, np.shape(pin)[0]):
        if scored_samples[i] < novelty_thresh:
            count += 1

    pout = np.zeros( (count, 3) )
    count = 0
    for i in range(0, np.shape(pin)[0]):
        if scored_samples[i] < novelty_thresh:
            pout[count, :] = pin[i, 0:3]
            count += 1

    return pout

def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)



parser = argparse.ArgumentParser(description="Novelty Checker")
parser.add_argument('--noveltythresh', type=float)
parser.add_argument('--decimate', type=int)
parser.add_argument('--datasetroot', type=dir_path)
parser.add_argument('--datasetname', type=str)
parser.add_argument('--resultsroot', type=dir_path)
parser.add_argument('--vizcamera', type=bool, default=False)
parser.add_argument('--colorext', type=str)
parser.add_argument('--frames', nargs='+', type=int)
args = parser.parse_args()

n_components = 1000
novelty_thresh = 0

decimate_factor = args.decimate
datasets_root = args.datasetroot
dataset_name = args.datasetname
results_root = args.resultsroot

results_path = os.path.join(results_root, dataset_name)
pathlib.Path(results_path).mkdir(parents=True, exist_ok=True)

K = np.eye(3)
K[0, 0] = 525.0/decimate_factor
K[1, 1] = 525.0/decimate_factor
K[0, 2] = 319.5/decimate_factor
K[1, 2] = 239.5/decimate_factor

# used in open3d visualization
O3D_K = np.array([[935.30743609,   0.,         959.5],
                  [0.,         935.30743609, 539.5],
                  [0.,           0.,           1.]])

W = (int)(640/decimate_factor)
H = (int)(480/decimate_factor)

rgb_glob_string = os.path.join(
    datasets_root, dataset_name + '-color/*.' + args.colorext)
cprint('rgb glob string %s' % (rgb_glob_string), 'green')
depth_glob_string = os.path.join(
    datasets_root, dataset_name + '-depth/*.png')
cprint('depth glob string %s' % (depth_glob_string), 'green')
traj_string = os.path.join(datasets_root, dataset_name + '-traj.log')
cprint('traj path %s' % (traj_string), 'green')

rgb_paths = sorted(glob.glob(rgb_glob_string))
depth_paths = sorted(glob.glob(depth_glob_string))
traj = read_log_trajectory(traj_string)

n_frames = len(rgb_paths)

print(len(depth_paths), n_frames)
assert(len(depth_paths) == n_frames)
assert(len(traj) == n_frames)

iu = ImageUtils(K)
viz = VisOpen3D()

frames = args.frames[0:-1]
for i, fr in enumerate(frames):
    print('i: ' + str(i))
    print('fr: ' + str(fr))

    # load the pointcloud and ground truth
    pcld_curr, _ = iu.generate_pcld_wf(traj[fr].pose, rgb_path=rgb_paths[fr],
                                       depth_path=depth_paths[fr],
                                       size=(W, H))

    # Next frame
    next_fr = args.frames[i+1]
    pcld_next_np, _ = iu.generate_pcld_wf(traj[next_fr].pose, rgb_path=rgb_paths[next_fr],
                                          depth_path=depth_paths[next_fr],
                                          size=(W, H))

    # Check if a GMM exists in the results
    gmm_filepath = results_root + '/' + str(args.frames[i]).zfill(5) + '.npz'
    if not os.path.isfile(gmm_filepath):

        # If not, create one and save
        print(gmm_filepath + ' doesnt exist...creating now');
        g = PythonGMM(n_components=n_components,
                      init_params='k-means++', random_state=0)
        g.fit(pcld_curr)
        sampled_points, _ = g.sample(n_samples=25000)
        means = g.means_
        precs_chol = g.precisions_cholesky_
        weights = g.weights_
        np.savez(gmm_filepath, means=means, precs_chol=precs_chol, weights=weights, sampled_points=sampled_points)

    # Load the data
    gmm_4d_data = np.load(gmm_filepath)

    # Get the 3d part
    gmm_3d_part = get3DPart(gmm_4d_data)

    # Calculate the novelty check
    scored_samples = sogmm_open3d_py.score_samples(pcld_next_np[:,0:3],
                                                   gmm_3d_part[0],
                                                   gmm_3d_part[1],
                                                   gmm_3d_part[2])

    novel_points = novelty_check(pcld_next_np, scored_samples, novelty_thresh)

    # Color all points red
    pcld_next_o3d = generate_red_pcld(novel_points)
    pcld_next_blue = generate_blue_pcld(pcld_next_np[:,0:3])
    pcld_curr_recon_green = generate_green_pcld(gmm_4d_data['sampled_points'][:,0:3])

    # add to the visualizer
    viz.add_geometry(np_to_o3d(pcld_curr))
    viz.add_geometry(pcld_curr_recon_green)
    viz.add_geometry(pcld_next_blue)
    viz.add_geometry(pcld_next_o3d)
    if args.vizcamera:
        viz.draw_camera(K, traj[fr].pose, W, H, color=[0.0, 0.0, 0.0])
        viz.update_view_point(O3D_K, np.linalg.inv(traj[fr].pose))
        viz.update_renderer()
        viz.poll_events()

    time.sleep(1.0)

    cprint('frame ' + str(i), 'green')

cprint('done', 'green')
viz.render()

import cv2
import argparse
import numpy as np
import time

from pipeline import utils
from pipeline.base_pipeline import BasePipeline


class VideoPipelineMK3(BasePipeline):
    def __init__(self, dir, save_debug_visualization=False):
        self.dir = dir
        self.save_debug_visualization = save_debug_visualization

        # Number of frames to skip between used frames (eg. 2 means using frames 1, 4, 7 and dropping 2, 3, 5, 6)
        self.frames_to_skip = 1

        # Number of frames for initial estimation
        self.num_frames_initial_estimation = 10

        # Number of frames between baseline reset. Hack to avoid further frames having no or few features
        self.feature_reset_rate = 10000

        # params for ShiTomasi corner detection
        self.feature_params = {
            "maxCorners": 200,
            "qualityLevel": 0.5,
            "minDistance": 15,
            "blockSize": 10
        }

        # Parameters for lucas kanade optical flow
        self.lk_params = {
            "winSize": (15, 15),
            "maxLevel": 3,
            "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        }

        self.image_size = None

        self.camera_matrix = np.array([[765.16859169, 0., 379.11876567],
                                       [0., 762.38664643, 497.22086655],
                                       [0., 0., 1.]])

        self.recover_pose_reconstruction_distance_threshold = 50
        self.find_essential_mat_threshold = 3
        self.find_essential_mat_prob = 0.98

        self.min_num_points_in_point_cloud = 10

        self.debug_colors = np.random.randint(0, 255, (self.feature_params['maxCorners'], 3))

    def run(self):
        # Start by finding the images
        file, filename = self._get_video(self.dir)

        Rs = [np.eye(3)]
        Ts = [np.zeros((3, 1))]
        tracks = []
        point_cloud = np.full((self.feature_params['maxCorners'], 3), None, dtype=np.float_)

        # Loop through frames (using generators)
        counter = 0
        for next_track_slice, next_track_slice_index_mask, is_new_feature_set in self._process_next_frame(file):
            counter += 1
            tracks += [next_track_slice]

            if is_new_feature_set:
                assert len(tracks) == 1, 'Resetting KLT features is not yet implemented'
                continue

            track_pair = np.array(tracks[-2:])

            R, T, new_points, new_point_indexes = self._calculate_pose(track_pair, Rs[-1], Ts[-1], point_cloud)

            # if cloud is empty and there are not enough new points, try next frame
            n_points_in_point_cloud = (~np.isnan(point_cloud)).any(axis=1).sum()
            if n_points_in_point_cloud == 0 and len(new_points) < self.min_num_points_in_point_cloud:
                # don't forget to reset track vector and drop old unused data
                tracks = [tracks[-1]]
                continue

            # merge new points with existing point cloud
            self._merge_points_into_cloud(point_cloud, new_points, new_point_indexes)

            Rs += [R]
            Ts += [T]

        # when all frames are processed, plot result
        utils.write_to_viz_file(self.camera_matrix, Rs, Ts, point_cloud)
        utils.call_viz()

    def _calculate_pose(self, track_pair, prev_R, prev_T, point_cloud):
        point_cloud_mask = (~np.isnan(point_cloud)).any(axis=1)
        n_points_in_point_cloud = point_cloud_mask.sum()

        # ------ STEP 1 ----------------------------------------------------------------------------------------
        # calculate R, T_un, points with 2 tracks (5 point + recover pose)
        R_rel, T_rel, new_points, new_point_indexes = self._get_pose_from_two_tracks(track_pair)
        # ------ END STEP 1 ------------------------------------------------------------------------------------

        assert n_points_in_point_cloud == 0 or n_points_in_point_cloud > self.min_num_points_in_point_cloud, \
            "Point cloud should have zero points or more than the minimum and not something in between"

        # if not enough points are in the cloud and track slice at the same time skip refine step
        if n_points_in_point_cloud > self.min_num_points_in_point_cloud:

            # ------ STEP 2 ----------------------------------------------------------------------------------------
            # first convert R and T from camera i to 0's perspective
            R, T = utils.compose_RTs(R_rel, T_rel, prev_R, prev_T)

            # then convert back to i's perspective but now referencing frame 0 and not frame i-1
            R = R.transpose()
            T = np.matmul(R, -T)

            # create new index mask based on existing point cloud's and newly created track's
            point_cloud_index_mask = np.arange(len(point_cloud))[point_cloud_mask]
            not_nan_mask = (~np.isnan(track_pair[1][point_cloud_index_mask])).any(axis=1)

            # refine R and T based on previous point cloud
            R, T = self._get_pose_from_points_and_projection(
                track_slice=track_pair[1][point_cloud_index_mask][not_nan_mask],
                points_3d=point_cloud[point_cloud_index_mask][not_nan_mask],
                R=R,
                T=T
            )
            # Result is in camera 0's coordinate system
            # ------ END STEP 2 ------------------------------------------------------------------------------------

            # ------ STEP 3 ----------------------------------------------------------------------------------------
            # recalculate points based on refined R and T
            new_points = self._reproject_tracks_to_3d(
                prev_R.transpose(),
                np.matmul(prev_R.transpose(), -prev_T),
                R.transpose(),
                np.matmul(R.transpose(), -T),
                track_pair)
            new_point_indexes = np.arange(len(new_points))
            # ------ END STEP 3 ------------------------------------------------------------------------------------
        else:
            R = R_rel
            T = T_rel

        return R, T, new_points, new_point_indexes

    def _merge_points_into_cloud(self, point_cloud, new_points, new_point_indexes):
        # check which points still have no data
        nan_mask = np.array(range(point_cloud.shape[0]))[np.isnan(point_cloud).any(axis=1)]

        # for those, replace nan by a value
        replace_mask = np.intersect1d(nan_mask, new_point_indexes)
        average_mask = np.setdiff1d(new_point_indexes, replace_mask)

        # TODO: there's a bug here!

        # for the others, do the average (results in exponential smoothing)
        # point_cloud[replace_mask] = new_points[replace_mask]
        point_cloud[replace_mask] = new_points[[item in replace_mask for item in new_point_indexes]]
        point_cloud[average_mask] = (point_cloud[average_mask] + new_points[[item in average_mask for item in new_point_indexes]]) / 2


    # =================== INTERNAL FUNCTIONS ===========================================================================


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dir',
                        help='Directory with image files for reconstructions')
    parser.add_argument('-sd', '--save_debug_visualization',
                        help='Save debug visualizations to files?', action='store_true', default=False)
    args = parser.parse_args()

    start = time.time()
    sfm = VideoPipelineMK3(**vars(args))
    sfm.run()
    elapsed = time.time() - start
    print("Elapsed {}".format(elapsed))

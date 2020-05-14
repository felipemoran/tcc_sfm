import cv2
import argparse
import numpy as np
import time

from pipeline import utils
from pipeline.base_pipeline import BasePipeline
from pipeline.bundle_adjuster import BundleAdjuster


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
            "blockSize": 10,
        }

        # Parameters for lucas kanade optical flow
        self.lk_params = {
            "winSize": (15, 15),
            "maxLevel": 2,
            "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        }

        self.image_size = None

        self.camera_matrix = np.array(
            [
                [765.16859169, 0.0, 379.11876567],
                [0.0, 762.38664643, 497.22086655],
                [0.0, 0.0, 1.0],
            ]
        )

        self.recover_pose_reconstruction_distance_threshold = 50
        self.find_essential_mat_threshold = 3
        self.find_essential_mat_prob = 0.98

        self.min_num_points_in_point_cloud = 10

        self.debug_colors = np.random.randint(
            0, 255, (self.feature_params["maxCorners"], 3)
        )

        self.bundle_adjuster = BundleAdjuster(self.camera_matrix, verbose=2)

    def run(self):
        # Start by finding the images
        file, filename = self._get_video(self.dir)

        Rs = [np.eye(3)]
        Ts = [np.zeros((3, 1))]
        tracks = []
        track_index_masks = []
        cloud = np.full((self.feature_params["maxCorners"], 3), None, dtype=np.float_)
        cloud_slice = None

        # Loop through frames (using generators)
        counter = 0
        for (
            next_track_slice,
            next_track_slice_mask,
            is_new_feature_set,
        ) in self._process_next_frame(file):
            tracks += [next_track_slice]

            counter += 1
            if is_new_feature_set:
                track_index_masks += [next_track_slice_mask]
                cloud_slice = cloud[: len(next_track_slice)]
                assert len(tracks) == 1, "Resetting KLT features is not yet implemented"
                continue

            track_pair = np.array(tracks[-2:])

            R, T, new_points, new_points_mask = self._calculate_pose(
                track_pair, Rs[-1], Ts[-1], cloud_slice
            )

            if R is None:
                # Last track slice wasn't good (not enough points) so let's drop it
                tracks = tracks[:-1]
                continue

            # if cloud is empty and there are not enough new points, try next frame
            n_points_in_point_cloud = (~utils.get_nan_mask(cloud_slice)).sum()
            if (
                n_points_in_point_cloud == 0
                and len(new_points) < self.min_num_points_in_point_cloud
            ):
                # don't forget to reset track vector and drop old unused data
                tracks = [tracks[-1]]
                continue

            # merge new points with existing point cloud
            self._merge_points_into_cloud(cloud_slice, new_points, new_points_mask)

            Rs += [R]
            Ts += [T]
            track_index_masks += [new_points_mask]

            assert len(Rs) == len(Ts) == len(tracks)

        #     # perform intermediate BA step
        #     optimized_cloud_slice, Rs, Ts = self.bundle_adjuster.run(
        #         cloud_slice, Rs, Ts, tracks, track_index_masks
        #     )
        #     cloud_slice[:] = optimized_cloud_slice
        #
        # # perform final BA step
        # cloud, Rs, Ts = self.bundle_adjuster.run(
        #     cloud, Rs, Ts, tracks, track_index_masks
        # )

        # when all frames are processed, plot result
        utils.write_to_viz_file(self.camera_matrix, Rs, Ts, cloud)
        utils.call_viz()

    def _calculate_pose(self, track_pair, prev_R, prev_T, point_cloud):
        point_cloud_mask = ~utils.get_nan_mask(point_cloud)
        n_points_in_point_cloud = point_cloud_mask.sum()

        # ------ STEP 1 ----------------------------------------------------------------------------------------
        # calculate R, T_un, points with 2 tracks (5 point + recover pose)
        R_rel, T_rel, points_3d, points_3d_mask = self._get_pose_from_two_tracks(
            track_pair
        )
        # ------ END STEP 1 ------------------------------------------------------------------------------------

        assert (
            n_points_in_point_cloud == 0
            or n_points_in_point_cloud > self.min_num_points_in_point_cloud
        ), "Point cloud should have zero points or more than the minimum and not something in between"

        # if not enough points are in the cloud and track slice at the same time skip refine step
        if (
            n_points_in_point_cloud > self.min_num_points_in_point_cloud
            and R_rel is not None
        ):

            # ------ STEP 2 ----------------------------------------------------------------------------------------
            # first convert R and T from camera i to 0's perspective
            R, T = utils.compose_RTs(R_rel, T_rel, prev_R, prev_T)

            # then convert back to i's perspective but now referencing frame 0 and not frame i-1
            R, T = R.transpose(), np.matmul(R.transpose(), -T)

            # create new mask based on existing point cloud's and newly created track's
            track_slice_mask = ~utils.get_nan_mask(track_pair[1])
            projection_mask = track_slice_mask & point_cloud_mask
            # not_nan_mask = ~utils.get_nan_mask(track_pair[1][point_cloud_index_mask])

            # refine R and T based on previous point cloud
            R, T = self._get_pose_from_points_and_projection(
                track_slice=track_pair[1][projection_mask],
                points_3d=point_cloud[projection_mask],
                R=R,
                T=T,
            )
            # Result is in camera 0's coordinate system
            # ------ END STEP 2 ------------------------------------------------------------------------------------

            # ------ STEP 3 ----------------------------------------------------------------------------------------
            # recalculate points based on refined R and T
            points_3d, points_3d_mask = self._reproject_tracks_to_3d(
                R_1=prev_R.transpose(),
                T_1=np.matmul(prev_R.transpose(), -prev_T),
                R_2=R.transpose(),
                T_2=np.matmul(R.transpose(), -T),
                tracks=track_pair,
            )
            # ------ END STEP 3 ------------------------------------------------------------------------------------
        else:
            R, T = R_rel, T_rel

        return R, T, points_3d, points_3d_mask

    def _merge_points_into_cloud(self, point_cloud, new_points, new_points_mask):
        # check which points still have no data
        nan_mask = np.isnan(point_cloud).any(axis=1)

        # for those, replace nan by a value
        replace_mask = nan_mask & new_points_mask
        average_mask = new_points_mask & ~replace_mask

        # for the others, do the average (results in exponential smoothing)
        point_cloud[replace_mask] = new_points[replace_mask]
        point_cloud[average_mask] = (
            point_cloud[average_mask] + new_points[average_mask]
        ) / 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", help="Directory with image files for reconstructions")
    parser.add_argument(
        "-sd",
        "--save_debug_visualization",
        help="Save debug visualizations to files?",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()

    start = time.time()
    sfm = VideoPipelineMK3(**vars(args))
    sfm.run()
    elapsed = time.time() - start
    print("Elapsed {}".format(elapsed))

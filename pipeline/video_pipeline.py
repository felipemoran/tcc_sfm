import cv2
import argparse
import numpy as np
import time

from pipeline import utils
from pipeline.base_pipeline import BasePipeline
from pipeline.bundle_adjuster import BundleAdjuster


class VideoPipeline(BasePipeline):
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
            "maxCorners": 100,
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
        self.find_essential_mat_prob = 0.99

        self.min_num_points_in_point_cloud = 10

        self.min_points_five_pt_algorithm = 6
        self.min_num_points_for_solvepnp = 6

        self.debug_colors = np.random.randint(
            0, 255, (self.feature_params["maxCorners"], 3)
        )

        self.bundle_adjuster = BundleAdjuster(self.camera_matrix, verbose=2)

        self.use_five_pt_algorithm = True
        self.use_solve_pnp = True
        self.reproject_tracks = True

        self.use_ba_at_end = True
        self.use_rolling_ba = True
        self.ba_window_length = 25
        self.ba_period = 10

        assert (
            self.use_five_pt_algorithm or self.use_solve_pnp
        ), "At least one algorithm between fiv-pt and solvepnp must be used"

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
            track_index_masks += [next_track_slice_mask]

            if is_new_feature_set:
                # track_index_masks += [next_track_slice_mask]
                cloud_slice = cloud[: len(next_track_slice)]
                assert len(tracks) == 1, "Resetting KLT features is not yet implemented"
                continue

            track_pair = np.array(tracks[-2:])

            R, T, new_points = self._calculate_pose(
                track_pair, Rs[-1], Ts[-1], cloud_slice
            )
            if self.reproject_tracks:
                # recalculate points based on refined R and T
                new_points = self._reproject_tracks_to_3d(
                    R_1=Rs[-1].transpose(),
                    T_1=np.matmul(Rs[-1].transpose(), -Ts[-1]),
                    R_2=R.transpose(),
                    T_2=np.matmul(R.transpose(), -T),
                    tracks=track_pair,
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
            if len(new_points) > 0:
                self._merge_points_into_cloud(cloud_slice, new_points)

            Rs += [R]
            Ts += [T]

            assert len(Rs) == len(Ts) == len(tracks)

            if self.use_rolling_ba:
                # perform intermediate BA step
                if counter % self.ba_period == 0:
                    bawl = self.ba_window_length
                    (
                        cloud_slice[:],
                        Rs[-bawl:],
                        Ts[-bawl:],
                    ) = self.bundle_adjuster.run(
                        cloud_slice,
                        Rs[-bawl:],
                        Ts[-bawl:],
                        tracks[-bawl:],
                        track_index_masks[-bawl:],
                    )
                # cloud_slice[:] = optimized_cloud_slice

        if self.use_ba_at_end:
            # perform final BA step
            cloud, Rs, Ts = self.bundle_adjuster.run(
                cloud_slice, Rs, Ts, tracks, track_index_masks
            )

        # when all frames are processed, plot result
        utils.write_to_viz_file(self.camera_matrix, Rs, Ts, cloud)
        utils.call_viz()

    def _calculate_pose(self, track_pair, prev_R, prev_T, point_cloud):
        point_cloud_mask = ~utils.get_nan_mask(point_cloud)
        n_points_in_point_cloud = point_cloud_mask.sum()
        track_pair_mask = ~utils.get_nan_mask(np.hstack(track_pair))

        R, T = None, None
        points_3d = np.empty((0, 3), dtype=np.float_)

        if self.use_five_pt_algorithm or n_points_in_point_cloud == 0:
            if track_pair_mask.sum() < self.min_points_five_pt_algorithm:
                return None, None, None

            R, T, points_3d = self._run_five_pt_algorithm(track_pair)

            # first convert R and T from camera i to 0's perspective
            points_3d = utils.translate_points_to_base_frame(prev_R, prev_T, points_3d)

            R, T = utils.compose_RTs(R, T, prev_R, prev_T)

        # create new mask based on existing point cloud's and newly created track's
        track_slice_mask = ~utils.get_nan_mask(track_pair[1])
        proj_mask = track_slice_mask & point_cloud_mask

        # Not enough points for reconstruction of pose
        if not self.use_solve_pnp or proj_mask.sum() < self.min_num_points_for_solvepnp:
            return R, T, points_3d

        if R is not None:
            R, T = utils.invert_reference_frame(R, T)

        # refine R and T based on previous point cloud
        R, T = self._run_solvepnp(
            track_slice=track_pair[1][proj_mask],
            points_3d=point_cloud[proj_mask],
            R=R,
            T=T,
        )
        # Result is in camera 0's coordinate system

        return R, T, points_3d

    def _merge_points_into_cloud(self, point_cloud, new_points):
        # check which points still have no data
        new_points_mask = ~utils.get_nan_mask(new_points)
        cloud_nan_mask = utils.get_nan_mask(point_cloud)

        # for those, replace nan by a value
        replace_mask = cloud_nan_mask & new_points_mask
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
    sfm = VideoPipeline(**vars(args))
    sfm.run()
    elapsed = time.time() - start
    print("Elapsed {}".format(elapsed))

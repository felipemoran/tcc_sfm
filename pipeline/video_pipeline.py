import argparse
import numpy as np
import time
import dacite

from ruamel.yaml import YAML
from pipeline import utils
from pipeline.base_pipeline import BasePipeline
from pipeline.bundle_adjuster import BundleAdjuster
from pipeline.config import VideoPipelineConfig


class VideoPipeline(BasePipeline):
    def __init__(
        self,
        dir: str,
        config: VideoPipelineConfig,
        display_klt_debug_frames: bool = False,
    ) -> None:
        self.dir = dir
        self.display_klt_debug_frames = display_klt_debug_frames
        self.config = config

        self.config.camera_matrix = np.array(self.config.camera_matrix)
        self.bundle_adjuster = BundleAdjuster(
            config=self.config.bundle_adjustment,
            camera_matrix=self.config.camera_matrix,
            verbose=2,
        )

    def run(self):
        # Start by finding the images
        file, filename = self._get_video(self.dir)

        Rs = [np.eye(3)]
        Ts = [np.zeros((3, 1))]
        tracks = []
        track_index_masks = []
        cloud = np.full(
            (self.config.klt.corner_selection.max_corners, 3),
            None,
            dtype=np.float_,
        )
        cloud_slice = None

        # Loop through frames (using generators)
        for frame_index, (track_slice, is_new_feature_set) in enumerate(
            self._process_next_frame(file)
        ):
            tracks += [track_slice]
            track_index_masks += [~utils.get_nan_mask(track_slice)]

            if is_new_feature_set:
                cloud_slice = cloud[: len(track_slice)]
                assert (
                    len(tracks) == 1
                ), "Resetting KLT features is not yet implemented"
                continue

            track_pair = np.array(tracks[-2:])

            R, T, new_points = self._calculate_pose(
                track_pair, Rs[-1], Ts[-1], cloud_slice
            )

            if R is None:
                # Last track slice wasn't good (not enough points) so let's drop it
                tracks = tracks[:-1]
                continue

            # merge new points with existing point cloud
            if len(new_points) > 0:
                self._merge_points_into_cloud(cloud_slice, new_points)

            Rs += [R]
            Ts += [T]

            assert len(Rs) == len(Ts) == len(tracks)

            cloud_slice[:], Rs, Ts = self.bundle_adjuster.run(
                cloud_slice, Rs, Ts, tracks, track_index_masks,
            )

        cloud_slice[:], Rs, Ts = self.bundle_adjuster.run(
            cloud_slice, Rs, Ts, tracks, track_index_masks, final_frame=True
        )

        # when all frames are processed, plot result
        utils.write_to_viz_file(self.config.camera_matrix, Rs, Ts, cloud)
        utils.call_viz()

    def _calculate_pose(self, track_pair, prev_R, prev_T, point_cloud):
        point_cloud_mask = ~utils.get_nan_mask(point_cloud)
        n_points_in_point_cloud = point_cloud_mask.sum()
        track_pair_mask = ~utils.get_nan_mask(np.hstack(track_pair))

        R, T = None, None
        points = np.empty((0, 3), dtype=np.float_)

        if self.config.use_five_pt_algorithm or n_points_in_point_cloud == 0:
            if (
                track_pair_mask.sum()
                < self.config.five_point_algorithm.min_number_of_points
            ):
                return None, None, None

            R, T, points = self._run_five_pt_algorithm(track_pair)

            # first convert R and T from camera i to 0's perspective
            points = utils.translate_points_to_base_frame(
                prev_R, prev_T, points
            )

            R, T = utils.compose_RTs(R, T, prev_R, prev_T)

        # create new mask based on existing point cloud's and newly created track's
        track_slice_mask = ~utils.get_nan_mask(track_pair[1])
        proj_mask = track_slice_mask & point_cloud_mask

        if (
            self.config.use_solve_pnp
            and proj_mask.sum() >= self.config.solvepnp.min_number_of_points
        ):

            R, T = utils.invert_reference_frame(R, T)

            # refine R and T based on previous point cloud
            R, T = self._run_solvepnp(
                track_slice=track_pair[1][proj_mask],
                points_3d=point_cloud[proj_mask],
                R=R,
                T=T,
            )
            # Result is in camera 0's coordinate system

        if self.config.use_reconstruct_tracks and R is not None:
            points = self._reproject_tracks_to_3d(
                R_1=prev_R.transpose(),
                T_1=np.matmul(prev_R.transpose(), -prev_T),
                R_2=R.transpose(),
                T_2=np.matmul(R.transpose(), -T),
                tracks=track_pair,
            )

        return R, T, points

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
    yaml = YAML()

    with open("config.yaml", "r") as f:
        config_raw = yaml.load(f)
    config = dacite.from_dict(data=config_raw, data_class=VideoPipelineConfig)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dir", help="Directory with image files for reconstructions"
    )
    parser.add_argument(
        "-sd",
        "--display_klt_debug_frames",
        help="Display KLT debug frames",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()

    start = time.time()
    sfm = VideoPipeline(**vars(args), config=config)
    sfm.run()
    elapsed = time.time() - start
    print("Elapsed {}".format(elapsed))

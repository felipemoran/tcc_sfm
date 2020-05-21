import argparse
import numpy as np
import time
import dacite
import cv2

from operator import itemgetter
from ruamel.yaml import YAML
from pipeline import utils
from pipeline.base_pipeline import BasePipeline
from pipeline.bundle_adjuster import BundleAdjuster
from pipeline.config import VideoPipelineConfig


class VideoPipeline(BasePipeline):
    def __init__(
        self,
        config: VideoPipelineConfig,
        display_klt_debug_frames: bool = False,
    ) -> None:
        self.display_klt_debug_frames = display_klt_debug_frames
        self.config = config

        self.config.camera_matrix = np.array(self.config.camera_matrix)
        self.bundle_adjuster = BundleAdjuster(
            config=self.config.bundle_adjustment,
            camera_matrix=self.config.camera_matrix,
            verbose=2,
        )

    def run(self, dir):
        file, _ = self._get_video(dir)
        track_generator = self._file_to_tracks(file)

        Rs, Ts, cloud, tracks = self._init_reconstruction(track_generator)

        Rs, Ts, cloud, tracks = self._reconstruct(
            track_generator, Rs, Ts, cloud, tracks
        )

        utils.write_to_viz_file(self.config.camera_matrix, Rs, Ts, cloud)
        utils.call_viz()

    def _init_reconstruction(self, track_generator):
        config = self.config.init

        Rs = None
        Ts = None
        cloud = None
        tracks = None

        for track_index, (track_slice, new_feature_set) in enumerate(
            track_generator
        ):
            if new_feature_set:
                Rs = []
                Ts = []
                tracks = []

            tracks += [track_slice]

            if config.method == "five_pt_algorithm":
                Rs, Ts, cloud, tracks = self._five_pt_init(Rs, Ts, tracks)
            elif config.method == "three_frame_combo":
                Rs, Ts, cloud, tracks = self._three_frame_init(
                    Rs, Ts, cloud, tracks
                )
            else:
                raise ValueError

            error = self._calculate_projection_error(Rs, Ts, cloud, tracks)
            print(f"Error: {error}")

            if error < config.error_threshold:
                return Rs, Ts, cloud, tracks
        else:
            raise Exception("Not enough frames for init phase")

    def _reconstruct(self, track_generator, Rs, Ts, cloud, tracks):
        for index, (track_slice, new_feature_set) in enumerate(track_generator):
            assert not new_feature_set, "Not yet implemented"

            tracks += [track_slice]
            track_pair = np.array(tracks[-2:])

            R, T, points = self._calculate_pose(
                track_pair, Rs[-1], Ts[-1], cloud
            )

            if R is None:
                self._trigger_klt_reset()
                break

            if points is not None:
                cloud = self._merge_points_into_cloud(cloud, points)

            Rs += [R]
            Ts += [T]

            Rs, Ts, cloud = self._run_ba(Rs, Ts, cloud, tracks)

        else:
            Rs, Ts, cloud = self._run_ba(Rs, Ts, cloud, tracks, True)

        return Rs, Ts, cloud, tracks

    def _calculate_pose(self, track_pair, prev_R, prev_T, cloud):
        if len(track_pair) == 1:
            return (*utils.init_rt(), None)

        assert len(track_pair) == 2

        R, T, points = None, None, None

        if self.config.use_five_pt_algorithm:
            R, T, points = self._run_five_pt_algorithm(
                track_pair, prev_R, prev_T
            )

        if self.config.use_solve_pnp:
            # refine R and T based on previous point cloud
            # result is in camera 0's coordinate system
            R, T = self._run_solvepnp(track_pair[1], cloud, R, T,)

        if self.config.use_reconstruct_tracks:
            points = self._reproject_tracks_to_3d(
                R_1=prev_R.transpose(),
                T_1=np.matmul(prev_R.transpose(), -prev_T),
                R_2=R.transpose(),
                T_2=np.matmul(R.transpose(), -T),
                tracks=track_pair,
            )

        return R, T, points

    def _merge_points_into_cloud(self, cloud, points):
        if cloud is None:
            cloud = np.full((len(points), 3), None, dtype=np.float_)

        # check which points still have no data
        points_not_nan_mask = ~utils.get_nan_mask(points)
        cloud_nan_mask = utils.get_nan_mask(cloud)

        # for those, replace nan by a value
        replace_mask = cloud_nan_mask & points_not_nan_mask
        average_mask = points_not_nan_mask & ~replace_mask

        # for the others, do the average (results in exponential smoothing)
        cloud[replace_mask] = points[replace_mask]
        # cloud[average_mask] = (cloud[average_mask] + points[average_mask]) / 2

        return cloud

    def _calculate_projection_error(self, Rs, Ts, cloud, tracks):
        if cloud is None:
            return float("inf")

        cloud_mask = ~utils.get_nan_mask(cloud)
        error = 0

        for index, (R, T, original_track) in enumerate(zip(Rs, Ts, tracks)):
            R_cam, T_cam = utils.invert_reference_frame(R, T)
            R_cam_vec = cv2.Rodrigues(R_cam)[0]

            projection_track = cv2.projectPoints(
                cloud[cloud_mask],
                R_cam_vec,
                T_cam,
                self.config.camera_matrix,
                None,
            )[0].squeeze()

            # filter out points not in both tracks
            original_track_mask = ~utils.get_nan_mask(original_track)
            projection_track_mask = ~utils.get_nan_mask(projection_track)
            mask = original_track_mask[cloud_mask] & projection_track_mask

            delta = original_track[cloud_mask][mask] - projection_track[mask]
            error += np.linalg.norm(delta, axis=1).mean()

        return error / (index + 1)

    def _trigger_klt_reset(self):
        raise NotImplementedError

    def _run_ba(self, Rs, Ts, cloud, tracks, final_frame=False):
        config = self.config.bundle_adjustment

        if (config.use_with_first_pair and len(Rs) == 2) or (
            config.use_at_end and final_frame
        ):
            Rs, Ts, cloud = self.bundle_adjuster.run(Rs, Ts, cloud, tracks,)

        elif (
            config.use_with_rolling_window
            and len(Rs) % config.rolling_window.period == 0
        ):
            method = config.rolling_window.method
            length = config.rolling_window.length
            step = config.rolling_window.step

            if method == "constant_step":
                ba_window_step = step
                ba_window_start = -(length - 1) * step - 1

                (
                    Rs[ba_window_start::ba_window_step],
                    Ts[ba_window_start::ba_window_step],
                    cloud,
                ) = self.bundle_adjuster.run(
                    Rs[ba_window_start::ba_window_step],
                    Ts[ba_window_start::ba_window_step],
                    cloud,
                    tracks[ba_window_start::ba_window_step],
                )
            elif method == "growing_step":
                indexes = [
                    item
                    for item in [
                        -int(i * (i + 1) / 2 + 1) for i in range(20, -1, -1)
                    ]
                    if -item <= len(Rs)
                ]

                R_opt, T_opt, cloud = self.bundle_adjuster.run(
                    itemgetter(*indexes)(Rs),
                    itemgetter(*indexes)(Ts),
                    cloud,
                    itemgetter(*indexes)(tracks),
                )

                for index, R, T in zip(indexes, R_opt, T_opt):
                    Rs[index] = R
                    Ts[index] = T

        return Rs, Ts, cloud

    def _five_pt_init(self, Rs, Ts, tracks):
        config = self.config.init.five_point
        if len(tracks) == 1:
            R, T = utils.init_rt()
            return [R], [T], None, tracks

        if len(tracks) > 2:
            if config.first_frame_fixed:
                Rs, Ts, tracks = Rs[:1], Ts[:1], [tracks[0], tracks[-1]]
            else:
                Rs, Ts, tracks = Rs[:1], Ts[:1], tracks[-2:]

        track_pair = np.array(tracks)
        R, T, points = self._run_five_pt_algorithm(track_pair, Rs[-1], Ts[-1])
        if R is None:
            self._trigger_klt_reset()

        points = self._reproject_tracks_to_3d(
            R_1=Rs[-1].transpose(),
            T_1=np.matmul(Rs[-1].transpose(), -Ts[-1]),
            R_2=R.transpose(),
            T_2=np.matmul(R.transpose(), -T),
            tracks=track_pair,
        )

        Rs += [R]
        Ts += [T]

        Rs, Ts, cloud = self._run_ba(Rs, Ts, points, tracks)

        return Rs, Ts, points, tracks

    def _three_frame_init(self, Rs, Ts, cloud, tracks):
        raise NotImplementedError


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
    sfm = VideoPipeline(
        display_klt_debug_frames=args.display_klt_debug_frames, config=config
    )
    sfm.run(args.dir)
    elapsed = time.time() - start
    print("Elapsed {}".format(elapsed))

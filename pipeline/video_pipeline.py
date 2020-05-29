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

        Rs, Ts, cloud, tracks, masks = self._init_reconstruction(
            track_generator
        )

        Rs, Ts, cloud, tracks = self._reconstruct(
            track_generator, Rs, Ts, cloud, tracks, masks
        )

        utils.write_to_viz_file(self.config.camera_matrix, Rs, Ts, cloud)
        utils.call_viz()

    def _init_reconstruction(self, track_generator):
        config = self.config.init

        Rs = []
        Ts = []
        cloud = None
        tracks = []
        masks = []

        for track_index, (track_slice, index_mask) in enumerate(
            track_generator
        ):
            tracks += [track_slice]
            masks += [index_mask]

            if config.method == "five_pt_algorithm":
                Rs, Ts, cloud, tracks, masks = self._five_pt_init(
                    Rs, Ts, tracks, masks
                )
            elif config.method == "three_frame_combo":
                Rs, Ts, cloud, tracks = self._three_frame_init(
                    Rs, Ts, cloud, tracks, masks
                )
            else:
                raise ValueError

            error = self._calculate_projection_error(
                Rs, Ts, cloud, tracks, masks
            )
            print(f"Error: {error}")

            if error < config.error_threshold:
                return Rs, Ts, cloud, tracks, masks
        else:
            raise Exception("Not enough frames for init phase")

    def _reconstruct(self, track_generator, Rs, Ts, cloud, tracks, masks):
        for index, (track_slice, index_mask) in enumerate(track_generator):

            tracks += [track_slice]
            masks += [index_mask]

            R, T, points, index_mask = self._calculate_pose(
                tracks, masks, Rs[-1], Ts[-1], cloud
            )

            if R is None:
                self._trigger_klt_reset()
                break

            if points is not None:
                cloud = self._add_points_to_cloud(cloud, points, index_mask)

            Rs += [R]
            Ts += [T]

            Rs, Ts, cloud = self._run_ba(Rs, Ts, cloud, tracks, masks)

        else:
            Rs, Ts, cloud = self._run_ba(Rs, Ts, cloud, tracks, masks, True)

        return Rs, Ts, cloud, tracks

    def _calculate_pose(self, tracks, masks, prev_R, prev_T, cloud):
        track_pair, pair_mask = utils.get_last_track_pair(tracks, masks)

        if len(track_pair) == 1:
            return (*utils.init_rt(), None)

        assert len(track_pair) == 2

        R, T, points, index_mask = None, None, None, None

        if self.config.use_five_pt_algorithm:
            R, T, points, bool_mask = self._run_five_pt_algorithm(
                track_pair, prev_R, prev_T
            )
            index_mask = pair_mask[bool_mask]

        if self.config.use_solve_pnp:
            # refine R and T based on previous point cloud
            # result is in camera 0's coordinate system
            R, T = self._run_solvepnp(tracks[-1], masks[-1], cloud, R, T)

        if self.config.use_reconstruct_tracks:
            points = self._reproject_tracks_to_3d(
                R_1=prev_R.transpose(),
                T_1=np.matmul(prev_R.transpose(), -prev_T),
                R_2=R.transpose(),
                T_2=np.matmul(R.transpose(), -T),
                tracks=track_pair,
            )
            index_mask = pair_mask

        return R, T, points, index_mask

    def _add_points_to_cloud(self, cloud, points, index_mask):
        if cloud is None:
            cloud = np.full((max(index_mask) * 2, 3), None, dtype=np.float_)

        cloud_mask = utils.get_not_nan_index_mask(cloud)
        new_points_mask = np.setdiff1d(index_mask, cloud_mask)

        if max(index_mask) > cloud.shape[0]:
            new_cloud = np.full((max(index_mask) * 2, 3), None, dtype=np.float_)
            new_cloud[cloud_mask] = cloud[cloud_mask]
            cloud = new_cloud

        cloud[new_points_mask] = points[np.isin(index_mask, new_points_mask)]

        return cloud

    def _calculate_projection_error(self, Rs, Ts, cloud, tracks, masks):
        # TODO: convert mask type

        if cloud is None:
            return float("inf")

        cloud_mask = utils.get_not_nan_index_mask(cloud)
        error = 0

        for index, (R, T, original_track, track_mask) in enumerate(
            zip(Rs, Ts, tracks, masks)
        ):
            intersection_mask = utils.get_intersection_mask(
                cloud_mask, track_mask
            )
            # track_bool_mask = [item in intersection_mask for item in track_mask]
            track_bool_mask = np.isin(track_mask, intersection_mask)

            R_cam, T_cam = utils.invert_reference_frame(R, T)
            R_cam_vec = cv2.Rodrigues(R_cam)[0]

            projection_track = cv2.projectPoints(
                cloud[intersection_mask],
                R_cam_vec,
                T_cam,
                self.config.camera_matrix,
                None,
            )[0].squeeze()

            delta = original_track[track_bool_mask] - projection_track
            error += np.linalg.norm(delta, axis=1).mean()

        return error / (index + 1)

    def _trigger_klt_reset(self):
        raise NotImplementedError

    def _run_ba(self, Rs, Ts, cloud, tracks, masks, final_frame=False):
        # TODO: convert mask type

        config = self.config.bundle_adjustment

        if (config.use_with_first_pair and len(Rs) == 2) or (
            config.use_at_end and final_frame
        ):
            Rs, Ts, cloud = self.bundle_adjuster.run(
                Rs, Ts, cloud, tracks, masks
            )

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
                    masks[ba_window_start::ba_window_step],
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
                    itemgetter(*indexes)(masks),
                )

                for index, R, T in zip(indexes, R_opt, T_opt):
                    Rs[index] = R
                    Ts[index] = T

        return Rs, Ts, cloud

    def _five_pt_init(self, Rs, Ts, tracks, masks):
        # TODO: convert mask type

        config = self.config.init.five_point

        if len(tracks) == 1:
            R, T = utils.init_rt()
            return [R], [T], None, tracks, masks

        if len(tracks) > 2:
            if config.first_frame_fixed:
                Rs, Ts, tracks, masks = (
                    Rs[:1],
                    Ts[:1],
                    [tracks[0], tracks[-1]],
                    [masks[0], masks[-1]],
                )
            else:
                Rs, Ts, tracks, masks = Rs[:1], Ts[:1], tracks[-2:], masks[-2:]

        track_pair, pair_mask = utils.get_last_track_pair(tracks, masks)

        R, T, points, bool_mask = self._run_five_pt_algorithm(
            track_pair, Rs[-1], Ts[-1]
        )
        if R is None:
            self._trigger_klt_reset()

        points = self._reproject_tracks_to_3d(
            R_1=Rs[-1].transpose(),
            T_1=np.matmul(Rs[-1].transpose(), -Ts[-1]),
            R_2=R.transpose(),
            T_2=np.matmul(R.transpose(), -T),
            tracks=track_pair,
        )

        cloud = utils.points_to_cloud(points=points, indexes=pair_mask)

        Rs += [R]
        Ts += [T]

        Rs, Ts, cloud = self._run_ba(Rs, Ts, cloud, tracks, masks)

        return Rs, Ts, cloud, tracks, masks

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

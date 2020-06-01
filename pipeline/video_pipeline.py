import argparse
import numpy as np
import time
import dacite
import cv2

from operator import itemgetter
from ruamel.yaml import YAML
from pipeline import utils
from pipeline import bundle_adjustment
from pipeline.reconstruction_algorithms import (
    calculate_projection,
    calculate_projection_error,
)
from pipeline.video_algorithms import get_video, klt_generator
from pipeline.init_algorithms import five_pt_init, three_frame_init
from pipeline.config import VideoPipelineConfig


class VideoPipeline:
    def __init__(
        self,
        config: VideoPipelineConfig,
        display_klt_debug_frames: bool = False,
    ) -> None:
        self.display_klt_debug_frames = display_klt_debug_frames
        self.config = config

        # self.config.camera_matrix = np.array(self.config.camera_matrix)

    def run(self, dir):
        file, _ = get_video(dir)
        track_generator = klt_generator(self.config.klt, file)

        Rs, Ts, cloud, tracks, masks = self._init_reconstruction(
            track_generator
        )

        Rs, Ts, cloud, tracks = self._reconstruct(
            track_generator, Rs, Ts, cloud, tracks, masks
        )

        utils.visualize(self.config.camera_matrix, Rs, Ts, cloud)

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
                Rs, Ts, cloud, tracks, masks = five_pt_init(
                    config.five_pt, Rs, Ts, tracks, masks
                )
            elif config.method == "three_frame_combo":
                Rs, Ts, cloud, tracks = three_frame_init(
                    Rs, Ts, cloud, tracks, masks
                )
            else:
                raise ValueError

            error = calculate_projection_error(
                config.camera_matrix, Rs, Ts, cloud, tracks, masks
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

            R, T, points, index_mask = calculate_projection(
                self.config, tracks, masks, Rs[-1], Ts[-1], cloud
            )

            if points is not None:
                cloud = utils.add_points_to_cloud(cloud, points, index_mask)

            Rs += [R]
            Ts += [T]

            Rs, Ts, cloud = self._run_ba(Rs, Ts, cloud, tracks, masks)

        else:
            Rs, Ts, cloud = self._run_ba(Rs, Ts, cloud, tracks, masks, True)

        return Rs, Ts, cloud, tracks

    def _run_ba(self, Rs, Ts, cloud, tracks, masks, final_frame=False):
        # TODO: convert mask type

        config = self.config.bundle_adjustment

        if config.use_at_end and final_frame:
            Rs, Ts, cloud = bundle_adjustment.run(
                config, Rs, Ts, cloud, tracks, masks
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
                ) = bundle_adjustment.run(
                    config,
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
                        -int(i * (i + 1) / 2 + 1)
                        for i in range(length - 1, -1, -1)
                    ]
                    if -item <= len(Rs)
                ]

                R_opt, T_opt, cloud = bundle_adjustment.run(
                    config,
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

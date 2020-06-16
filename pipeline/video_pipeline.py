import argparse
import itertools
from itertools import product

import numpy as np
import time
import dacite
import cv2

from operator import itemgetter
from ruamel.yaml import YAML
from pipeline import utils
from pipeline import bundle_adjustment
from pipeline.errors import EndOfFileError
from pipeline.reconstruction_algorithms import (
    calculate_projection,
    calculate_projection_errors,
    solve_pnp,
)
from pipeline.video_algorithms import get_video, klt_generator
from pipeline.init_algorithms import five_pt_init
from pipeline.config import VideoPipelineConfig


class VideoPipeline:
    def __init__(self, config: VideoPipelineConfig,) -> None:
        self.config = config

        # self.config.camera_matrix = np.array(self.config.camera_matrix)

    def run(self):
        track_generator = self._setup(self.config.file_path)

        (
            Rs,
            Ts,
            cloud,
            tracks,
            masks,
            track_generator,
        ) = self._init_reconstruction(track_generator)

        Rs, Ts, cloud, tracks, masks = self._reconstruct(
            track_generator, Rs, Ts, cloud, tracks, masks
        )

        return Rs, Ts, cloud

    def _setup(self, dir):
        file, _ = get_video(dir)
        track_generator = klt_generator(self.config.klt, file)
        return track_generator

    def _init_reconstruction(self, track_generator):
        """

        @param track_generator: teste
        @return: return description
        """
        config = self.config.init

        tracks = []
        masks = []

        dropped_tracks = 0

        for index, (track_slice, mask) in enumerate(track_generator):
            tracks += [track_slice]
            masks += [mask]

            if (
                len(tracks)
                < config.num_reconstruction_frames
                + config.num_error_calculation_frames
            ):
                continue

            # TODO: call full reconstruction with K frames
            Rs, Ts, cloud, rec_tracks, rec_masks = self._reconstruct(
                zip(
                    tracks[: config.num_reconstruction_frames],
                    masks[: config.num_reconstruction_frames],
                )
            )

            # TODO: call error calculation with reconstruction and P frames
            error, err_tracks, err_masks = self._calculate_error(
                zip(
                    tracks[-config.num_error_calculation_frames :],
                    masks[-config.num_error_calculation_frames :],
                ),
                cloud,
            )

            print(
                f"{config.num_reconstruction_frames},"
                f"{config.num_error_calculation_frames},"
                f"{dropped_tracks},"
                f"{error}"
            )

            # exit init or or drop first track/mask
            if error > config.error_threshold:
                # drop first track ank mask and rerun the process
                tracks.pop(0)
                masks.pop(0)
                dropped_tracks += 1
            else:
                # add tracks used for error calculation back to track generator
                track_generator = itertools.chain(
                    zip(err_tracks, err_masks), track_generator
                )

                return Rs, Ts, cloud, rec_tracks, rec_masks, track_generator
        else:
            raise EndOfFileError("Not enough frames for init phase")

    def _reconstruct(
        self,
        track_generator,
        Rs=None,
        Ts=None,
        cloud=None,
        tracks=None,
        masks=None,
    ):
        if tracks is None:
            tracks, masks = [], []

        for index, (track_slice, index_mask) in enumerate(track_generator):
            tracks += [track_slice]
            masks += [index_mask]

            if cloud is None:
                Rs, Ts, cloud = five_pt_init(self.config, tracks, masks)
                continue

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

        return Rs, Ts, cloud, tracks, masks

    def _calculate_error(
        self, track_generator, cloud,
    ):
        errors, Rs, Ts, tracks, masks = [], [], [], [], []

        for index, (track, mask) in enumerate(track_generator):
            tracks += [track]
            masks += [mask]

            R, T = solve_pnp(self.config.solve_pnp, track, mask, cloud)
            Rs += [R]
            Ts += [T]

        errors = calculate_projection_errors(
            self.config.camera_matrix, Rs, Ts, cloud, tracks, masks
        )

        mean_error = np.array(
            [frame_errors.mean() for frame_errors in errors]
        ).mean()

        return mean_error, tracks, masks

    def _run_ba(self, Rs, Ts, cloud, tracks, masks, final_frame=False):

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

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
    calculate_projection_error,
    solve_pnp,
)
from pipeline.utils import ErrorMetric
from pipeline.video_algorithms import get_video, klt_generator
from pipeline.init_algorithms import five_pt_init
from pipeline.config import VideoPipelineConfig


class VideoPipeline:
    def __init__(self, config: VideoPipelineConfig,) -> None:
        self.config = config

        self._first_reconstruction_frame_number = None

        # self.config.camera_matrix = np.array(self.config.camera_matrix)

    def run(self):
        track_generator = self._setup(self.config.file_path)

        (
            Rs,
            Ts,
            cloud,
            tracks,
            masks,
            frame_numbers,
            track_generator,
            init_errors,
        ) = self._init_reconstruction(track_generator)

        if cloud is None:
            # init faild
            return None, None, None, init_errors, [], []

        (
            Rs,
            Ts,
            cloud,
            tracks,
            masks,
            frame_numbers,
            online_errors,
            post_errors,
        ) = self._reconstruct(
            track_generator, Rs, Ts, cloud, tracks, masks, frame_numbers
        )

        return Rs, Ts, cloud, init_errors, online_errors, post_errors

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
        frame_numbers = []

        init_errors = []

        dropped_tracks = 0

        for frame_number, track_slice, mask in track_generator:
            tracks += [track_slice]
            masks += [mask]
            frame_numbers += [frame_number]

            if (
                len(tracks)
                < config.num_reconstruction_frames
                + config.num_error_calculation_frames
            ):
                continue

            # update number of supposed first frame for reconstruction
            self._first_reconstruction_frame_number = frame_numbers[0]

            reconstruction = self._reconstruct(
                zip(
                    frame_numbers[: config.num_reconstruction_frames],
                    tracks[: config.num_reconstruction_frames],
                    masks[: config.num_reconstruction_frames],
                ),
                is_init=True,
            )
            cloud = reconstruction[2]

            # call error calculation with reconstruction and P frames
            error = self._calculate_init_error(
                tracks[-config.num_error_calculation_frames :],
                masks[-config.num_error_calculation_frames :],
                frame_numbers[-config.num_error_calculation_frames :],
                cloud,
            )
            init_errors += [error]

            # print(
            #     f"{self.config.bundle_adjustment.use_at_end},"
            #     f"{self.config.bundle_adjustment.use_with_rolling_window},"
            #     f"{self.config.bundle_adjustment.rolling_window.method},"
            #     f"{self.config.synthetic_config.noise_covariance},"
            #     f"{config.num_reconstruction_frames},"
            #     f"{config.num_error_calculation_frames},"
            #     f"{dropped_tracks},"
            #     f"{error}"
            # )

            # exit init or or drop first track/mask
            if error.projection > config.error_threshold:
                # drop first track ank mask and rerun the process
                tracks.pop(0)
                masks.pop(0)
                frame_numbers.pop(0)
                dropped_tracks += 1
            else:
                # add tracks used for error calculation back to track generator
                track_generator = itertools.chain(
                    zip(
                        frame_numbers[-config.num_error_calculation_frames :],
                        tracks[-config.num_error_calculation_frames :],
                        masks[-config.num_error_calculation_frames :],
                    ),
                    track_generator,
                )

                return reconstruction[:6] + (track_generator, init_errors)
        else:
            return (None,) * 7 + (init_errors,)

    def _reconstruct(
        self,
        track_generator,
        Rs=None,
        Ts=None,
        cloud=None,
        tracks=None,
        masks=None,
        frame_numbers=None,
        is_init=False,
    ):

        if tracks is None:
            tracks, masks, frame_numbers = [], [], []

        # Before processing any frames, calculate error metrics for current frames
        if not is_init and self.config.error_calculation.online_calculation:
            online_errors = self._calculate_reconstruction_errors_from_history(
                Rs, Ts, cloud, tracks, masks, frame_numbers
            )
        else:
            online_errors = []

        for frame_number, track_slice, index_mask in track_generator:
            tracks += [track_slice]
            masks += [index_mask]
            frame_numbers += [frame_number]

            # Init cloud
            if cloud is None:
                Rs, Ts, cloud = five_pt_init(self.config, tracks, masks)
                continue

            # Reconstruct
            R, T, points, index_mask = calculate_projection(
                self.config, tracks, masks, Rs[-1], Ts[-1], cloud
            )

            # Add new points to cloud
            if points is not None:
                cloud = utils.add_points_to_cloud(cloud, points, index_mask)

            # Save camera pose
            Rs += [R]
            Ts += [T]

            # Run optimizations
            Rs, Ts, cloud = self._run_ba(Rs, Ts, cloud, tracks, masks)

            # Calculate and store error metrics
            if is_init or not self.config.error_calculation.online_calculation:
                continue

            online_errors += [
                self._calculate_reconstruction_error(
                    *self._select_reconstruction_error_data(
                        Rs, Ts, tracks, masks, frame_numbers
                    ),
                    cloud,
                )
            ]

        # Optimize at end, but only if it's not in init phase
        if not is_init:
            Rs, Ts, cloud = self._run_ba(Rs, Ts, cloud, tracks, masks, True)

        if not is_init and self.config.error_calculation.post_calculation:
            post_errors = self._calculate_reconstruction_errors_from_history(
                Rs, Ts, cloud, tracks, masks, frame_numbers
            )
        else:
            post_errors = []

        return (
            Rs,
            Ts,
            cloud,
            tracks,
            masks,
            frame_numbers,
            online_errors,
            post_errors,
        )

    def _calculate_init_error(self, tracks, masks, frame_numbers, cloud):
        """
        Calculates the projection error for a set of frames given some init
         conditions.

        It calculates the error by first calculating the expected rotation and
        translation followed by the resulting 2D projection of this pose and
        then comparing it with the original track slice.

        :param tracks: list of 2D feature vectors. Each vector has the shape Dx2
        :param masks: list of index masks for each feature vector. Indexes refer to the position of the item in the cloud
        :param frame_numbers: list of indexes for each track in tracks
        :param cloud: actual point cloud
        :return: mean error
        """
        # Rs and Ts are only used in the synthetic pipeline

        errors, Rs, Ts, = [], [], []

        for frame_number, track, mask in zip(frame_numbers, tracks, masks):
            R, T = solve_pnp(self.config.solve_pnp, track, mask, cloud)
            Rs += [R]
            Ts += [T]

        # error = calculate_projection_error(
        #     self.config.camera_matrix, Rs, Ts, cloud, tracks, masks, mean=True
        # )

        error = self._calculate_reconstruction_error(
            Rs, Ts, tracks, masks, frame_numbers, cloud
        )

        return error

    def _calculate_reconstruction_error(
        self, Rs, Ts, tracks, masks, frame_numbers, cloud
    ):
        """
        
        :param Rs: list of R matrices
        :param Ts: list of T vectors
        :param tracks: list of 2D feature vectors. Each vector has the shape Dx2
        :param masks: list of index masks for each feature vector. Indexes refer to the position of the item in the cloud
        :param frame_numbers: list of indexes for each track in tracks
        :param cloud: point cloud with N points as a ndarray with shape Nx3
        :return:
        """

        projection_error = calculate_projection_error(
            self.config.camera_matrix, Rs, Ts, cloud, tracks, masks, mean=True
        )

        error = ErrorMetric(
            frame_numbers[-1], projection_error, np.nan, np.nan, np.nan
        )

        return error

    def _select_reconstruction_error_data(
        self, Rs, Ts, tracks, masks, frame_numbers
    ):
        """
        Selects data to be used on reconstruction error calculation. Returns
        slices of inputs.

        :param Rs: list of R matrices
        :param Ts: list of T vectors
        :param tracks: list of 2D feature vectors. Each vector has the shape Dx2
        :param masks: list of index masks for each feature vector. Indexes refer to the position of the item in the cloud
        :param frame_numbers: list of indexes for each track in tracks
        :return: slices of inputs with items to be used for calculation
        """
        if len(Rs) % self.config.error_calculation.period != 0:
            return [] * 5

        error_window = self.config.error_calculation.window_length

        return_slices = (
            Rs[-error_window:],
            Ts[-error_window:],
            tracks[-error_window:],
            masks[-error_window:],
            frame_numbers[-error_window:],
        )

        return return_slices

    def _calculate_reconstruction_errors_from_history(
        self, Rs, Ts, cloud, tracks, masks, frame_numbers
    ):
        """
        Similar to calculate_reconstruction_error but it calculates errors
        after all reconstruction.

        :param Rs:
        :param Ts:
        :param cloud:
        :param tracks:
        :param masks:
        :param frame_numbers:
        :return:
        """
        assert (
            len(Rs)
            == len(Ts)
            == len(tracks)
            == len(masks)
            == len(frame_numbers)
        )

        errors = []
        for i in range(1, len(Rs) + 1):
            errors += [
                self._calculate_reconstruction_error(
                    *self._select_reconstruction_error_data(
                        Rs[:i], Ts[:i], tracks[:i], masks[:i], frame_numbers[:i]
                    ),
                    cloud,
                )
            ]

        return errors

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

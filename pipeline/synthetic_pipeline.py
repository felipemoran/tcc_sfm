from itertools import product

import cv2
import itertools
import numpy as np

from math import pi, cos, sin, acos, degrees
from numpy.linalg import norm
from pipeline import utils
from pipeline.reconstruction_algorithms import calculate_projection_error
from pipeline.utils import ErrorMetric
from pipeline.video_pipeline import VideoPipeline
from pipeline.config import VideoPipelineConfig
from ruamel.yaml import YAML
import dacite

np.set_printoptions(3, suppress=True)

TYPE_CALIBRATION_MATRIX = 0
TYPE_CAMERA = 1
TYPE_POINT = 2


class SyntheticPipeline(VideoPipeline):
    """
    Class that inherist from VideoPipeline that allows for the generation and injection of
    synthetic data into the reconstruction pipeline for debugging and development purposes
    """

    def __init__(self, config):
        super().__init__(config)

        self.synthetic_case = None
        self.original_points = None
        self.original_Rs = None
        self.original_Ts = None

    def _setup(self, file_path):
        self.synthetic_case = int(file_path)
        return self._synthetic_generator()

    def _synthetic_generator(self):
        """
        Replaces the original generator yielding synthetic data instead of real data

        :return: yields track a track slice containing synthetic features and their corresponding indexes
        """
        config = self.config.synthetic_config

        number_of_cameras = config.number_of_cameras

        points_3d = self._get_synthetic_points()

        Rs = self._get_synthetic_camera_rotations()[:number_of_cameras]
        Ts = self._get_synthetic_camera_translations()[:number_of_cameras]

        self.original_points = points_3d
        self.original_Rs = Rs
        self.original_Ts = Ts

        for index, (R, T) in enumerate(zip(Rs, Ts)):
            # convert to the camera base, important!
            R_cam, T_cam = utils.invert_reference_frame(R, T)
            R_cam_vec = cv2.Rodrigues(R_cam)[0]

            track_slice = cv2.projectPoints(
                points_3d, R_cam_vec, T_cam, self.config.camera_matrix, None
            )[0].squeeze()

            noise = np.random.normal(
                loc=0.0, scale=config.noise_covariance, size=track_slice.shape
            )

            track_slice += noise

            slice_mask = (track_slice > 0).all(axis=1)

            # drop = np.arange(1, 4) * 4 + index
            # drop_bool = np.full(slice_mask.shape, True)
            # drop_bool[drop] = False
            # slice_mask = slice_mask & drop_bool

            index_mask = np.arange(len(points_3d))[slice_mask]

            track_slice = track_slice[slice_mask]

            yield index, track_slice, index_mask

    def _get_synthetic_points(self):
        """
        Returns synthetic 3D points
        """
        # points_3d = np.array(
        #     list(itertools.product([9, 10, 11], [4, 5, 6], [-1, 0, 1])), dtype=np.float_
        # )
        # points_3d = np.array(
        #     list(itertools.product([9, 11], [4, 6], [-1, 1])), dtype=np.float_
        # )

        if self.synthetic_case in (1, 2):
            points_3d = np.array(
                list(itertools.product([8, 9, 10, 11, 12], [4, 5, 6], [0]))
                + list(itertools.product([8, 9, 10], [4, 5], [1]))
                + list(itertools.product([8], [4], [2])),
                dtype=np.float_,
            )

        elif self.synthetic_case == 3:
            config = self.config.synthetic_config.case_3
            step = config.step_size

            x_points = config.x_points
            y_points = config.y_points
            z_points = config.z_points

            x_threshold = (x_points - 1) / 2 * step
            y_threshold = (y_points - 1) / 2 * step
            z_threshold = (z_points - 1) / 2 * step

            points = []
            for x, y, z in product(
                np.arange(-x_threshold, x_threshold + step / 2, step),
                np.arange(-y_threshold, y_threshold + step / 2, step),
                np.arange(-z_threshold, z_threshold + step / 2, step),
            ):
                # if abs((abs(x) + abs(y) + abs(z)) - threshold) > step / 10:
                #     continue
                if (
                    abs(abs(x) - x_threshold) > 0.001
                    and abs(abs(y) - y_threshold) > 0.001
                    and abs(abs(z) - z_threshold) > 0.001
                ):
                    continue

                points += [(x, y, z)]
            points_3d = np.array(list(set(points)))

        return points_3d

    def _get_synthetic_camera_rotations(self):
        """
        Returns rotation matrices of synthetic cameras in the global reference frame
        """
        # matrizes de rotação para o posicionamento das cameras
        r1 = cv2.Rodrigues(np.array([-pi / 2, 0.0, 0.0]))[0]
        r2 = cv2.Rodrigues(np.array([0, 0, -pi / 4]))[0]
        r3 = cv2.Rodrigues(np.array([0, -pi / 2, 0]))[0]

        if self.synthetic_case in (1, 2):
            # vetores de rotação das cameras na base global
            Rs = np.array(
                [
                    r1,
                    np.matmul(r1, r3),
                    np.matmul(r1, np.matmul(r3, np.matmul(r3, r2))),
                    np.matmul(r1, np.matmul(r3, np.matmul(r3, r3))),
                    np.matmul(r1, r1),
                ]
            )
        else:
            config = self.config.synthetic_config.case_3
            number_of_cameras = config.number_of_cameras
            delta_angle = 2 * pi / number_of_cameras

            Rs = np.array(
                [
                    r1.dot(cv2.Rodrigues(np.array([0, -delta_angle * i, 0]))[0])
                    for i in range(number_of_cameras)
                ]
            )

        return Rs

    def _get_synthetic_camera_translations(self):
        """
        Returns translation vectors of synthetic cameras in the global reference frame
        :return:
        """
        # vetores de translação das câmeras na base global
        if self.synthetic_case == 1:
            Ts = np.array(
                [[10, 0, 0], [15, 5, 0], [10, 10, 0], [5, 5, 0], [10, 5, 5],],
                dtype=np.float_,
            )
        elif self.synthetic_case == 2:
            Ts = np.array(
                [[10, 0, 0], [20, 5, 0], [10, 12.5, 0], [5, 5, 0], [10, 5, 5],],
                dtype=np.float_,
            )
        elif self.synthetic_case == 3:
            config = self.config.synthetic_config.case_3
            number_of_cameras = config.number_of_cameras
            radius = config.radius

            delta_angle = 2 * pi / number_of_cameras

            Ts = np.array(
                [
                    np.array(
                        [
                            radius * cos(-pi / 2 + delta_angle * i),
                            radius * sin(-pi / 2 + delta_angle * i),
                            0,
                        ]
                    ).reshape((3, 1))
                    for i in range(number_of_cameras)
                ]
            )

        else:
            raise ValueError
        return Ts

    def _calculate_reconstruction_error(
        self, Rs, Ts, tracks, masks, frame_numbers, cloud
    ):
        """
        Overwrite of superclass function that calculates camera orientation and
        position errors, and 3D point errors other than reprojection error. Results
        are structured in a struct as described below.

        :param Rs: list of R matrices
        :param Ts: list of T vectors
        :param tracks: list of 2D feature vectors. Each vector has the shape Dx2
        :param masks: list of index masks for each feature vector. Indexes refer to the position of the item in the cloud
        :param frame_numbers: list of indexes for each track in tracks
        :param cloud: point cloud with N points as a ndarray with shape Nx3
        :return:
        """
        error_metric = ErrorMetric(frame_numbers[-1], *[np.nan] * 4)

        projection_error = calculate_projection_error(
            self.config.camera_matrix, Rs, Ts, cloud, tracks, masks, mean=True
        )
        error_metric.projection = projection_error

        if self.synthetic_case != 3:
            return error_metric

        assert self._first_reconstruction_frame_number is not None

        config = self.config.synthetic_config.case_3

        errors = [[], [], []]

        calib_R = self.original_Rs[self._first_reconstruction_frame_number]
        calib_T = self.original_Ts[self._first_reconstruction_frame_number]

        scale = sin(2 * pi / (config.number_of_cameras * 2)) * 2 * config.radius

        for frame_number, track, mask, R, T in zip(
            frame_numbers, tracks, masks, Rs, Ts
        ):
            orig_R = self.original_Rs[frame_number]
            orig_T = self.original_Ts[frame_number]

            # calibrate R and T
            R = calib_R.dot(R)
            T = (calib_R.dot(T)) * scale + calib_T

            angle_offset = degrees(
                abs(acos((np.trace(R.transpose().dot(orig_R)) - 1) / 2))
            )

            position_offset = np.linalg.norm(T - orig_T)

            errors[0] += [angle_offset]
            errors[1] += [position_offset]

        for original_point, point in zip(self.original_points, cloud):
            if np.isnan(point).any():
                continue

            point = (
                (calib_R.dot(point.reshape((3, 1)))) * scale + calib_T
            ).squeeze()

            offset = np.linalg.norm(original_point - point)
            errors[2] += [offset]

        mean_error = [np.array(errors[i]).mean() for i in range(len(errors))]

        error_metric.cam_orientation = mean_error[0]
        error_metric.cam_position = mean_error[1]
        error_metric.point_position = mean_error[2]

        return error_metric

"""
Structure from Motion bundle adjusment module.

Based on: https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html

Authors: Caio Cancian, Felipe Moran

Version: 1.0.0

"""

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from config import BundleAdjustmentConfig
import cv2
from pipeline import utils


class BundleAdjuster:
    """Bundle adjusment base class.

    Performs end-to-end bundle adjusment to Structure from Motion problems.
    """

    def __init__(
        self,
        config: BundleAdjustmentConfig,
        camera_matrix,
        tol=1e-4,
        method="trf",
        verbose=0,
    ):
        self.config = config

        # Basic optimization attributes
        self.tol = tol
        self.method = method
        self.verbose = verbose
        self.camera_matrix = camera_matrix

        # Attributes to store resulting optimization parameters
        self.optimized_points = None
        self.optimized_cameras = None

    def run(self, point_cloud, Rs, Ts, tracks, track_masks, final_frame=False):
        run_ba = False
        ba_window_start = 0

        if self.config.use_with_first_pair and len(Rs) == 2:
            run_ba = True

        if (
            self.config.use_with_rolling_window
            and len(Rs) % self.config.rolling_window.period == 0
        ):
            run_ba = True
            ba_window_start = -self.config.rolling_window.length

        if self.config.use_at_end and final_frame:
            run_ba = True

        if not run_ba:
            return point_cloud, Rs, Ts

        (
            camera_params,
            points_3d,
            points_2d,
            camera_indexes,
            point_indexes,
        ) = self._prepare_optimization_input(
            point_cloud,
            Rs[ba_window_start:],
            Ts[ba_window_start:],
            tracks[ba_window_start:],
            track_masks[ba_window_start:],
        )

        # Optimize
        optimized_cameras, optimized_points = self.optimize(
            camera_params=camera_params,
            points_3d=points_3d,
            points_2d=points_2d,
            camera_indices=camera_indexes,
            point_indices=point_indexes,
        )
        (
            point_cloud[:],
            Rs[ba_window_start:],
            Ts[ba_window_start:],
        ) = self._parse_optimization_result(
            point_cloud=point_cloud,
            optimized_cameras=optimized_cameras,
            optimized_points=optimized_points,
        )

        return point_cloud, Rs, Ts

    def _prepare_optimization_input(self, point_cloud, Rs, Ts, tracks, track_masks):
        assert len(Rs) == len(Ts) == len(tracks) == len(track_masks)

        camera_params = []
        points_2d = np.empty((0, 2), dtype=np.float_)
        camera_indexes = np.empty((0,), dtype=int)
        point_indexes = np.empty((0,), dtype=int)

        for R, T in zip(Rs, Ts):
            R, T = utils.invert_reference_frame(R, T)
            camera_params += [np.vstack((cv2.Rodrigues(R)[0], T)).reshape(-1)]
        camera_params = np.array(camera_params, dtype=np.float_)
        assert camera_params.shape == (len(Rs), 6)

        cloud_not_nan_mask = ~utils.get_nan_mask(point_cloud)

        for index, (track, track_mask) in enumerate(zip(tracks, track_masks)):
            mask = track_mask & cloud_not_nan_mask

            camera_indexes = np.append(camera_indexes, np.full(mask.sum(), index))
            point_indexes = np.append(
                point_indexes,
                np.arange(cloud_not_nan_mask.sum())[track_mask[cloud_not_nan_mask]],
            )
            points_2d = np.vstack((points_2d, track[mask]))

        assert len(camera_indexes) == len(point_indexes) == len(points_2d)

        points_3d = point_cloud[cloud_not_nan_mask]

        return camera_params, points_3d, points_2d, camera_indexes, point_indexes

    def _parse_optimization_result(
        self, point_cloud, optimized_cameras, optimized_points
    ):
        # Convert back to pipeline style from BA style
        Rs = []
        Ts = []

        not_nan_mask = ~utils.get_nan_mask(point_cloud)
        point_cloud[not_nan_mask] = optimized_points

        for camera in optimized_cameras:
            R = cv2.Rodrigues(camera[:3])[0]
            T = camera[3:].reshape((3, -1))
            R, T = utils.invert_reference_frame(R, T)
            Rs += [R]
            Ts += [T]

        return point_cloud, Rs, Ts

    def _rotate(self, points, rot_vecs):
        """Rotate 3D points by given rotation vectors.

        Rodrigues' rotation formula is used.
        See https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
        for details
        """
        theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
        with np.errstate(invalid="ignore"):
            v = rot_vecs / theta
            v = np.nan_to_num(v)
        dot = np.sum(points * v, axis=1)[:, np.newaxis]
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        rot = (
            cos_theta * points
            + sin_theta * np.cross(v, points)
            + dot * (1 - cos_theta) * v
        )
        return rot

    def _project(self, points, camera_params):
        """Convert 3-D points to 2-D by projecting onto images.

        No radial distortion is considered.
        """

        # Rotate and translate
        points_proj = self._rotate(points, camera_params[:, :3])
        points_proj += camera_params[:, 3:6]

        # Divide by scale
        points_proj = points_proj[:, :2] / points_proj[:, 2, np.newaxis]

        # Set camera calibration params
        fx = self.camera_matrix[0, 0]
        cx = self.camera_matrix[0, 2]
        fy = self.camera_matrix[1, 1]
        cy = self.camera_matrix[1, 2]

        # Apply projection formula WITHOUT radial distortion
        points_proj[:, 0] = points_proj[:, 0] * fx + cx
        points_proj[:, 1] = points_proj[:, 1] * fy + cy

        return points_proj

    def _objective_function(
        self, params, n_cameras, n_points, camera_indices, point_indices, points_2d
    ):
        """Compute residuals.

        `params` contains camera parameters and 3-D coordinates.
        """
        camera_params = params[: n_cameras * 6].reshape((n_cameras, 6))
        points_3d = params[n_cameras * 6 :].reshape((n_points, 3))
        points_proj = self._project(
            points_3d[point_indices], camera_params[camera_indices]
        )
        return (points_proj - points_2d).ravel()

    def _bundle_adjustment_sparsity(
        self, n_cameras, n_points, camera_indices, point_indices
    ):
        """Build optimization sparse matrix."""

        m = camera_indices.size * 2
        n = n_cameras * 6 + n_points * 3
        A = lil_matrix((m, n), dtype=int)

        i = np.arange(camera_indices.size)
        for s in range(6):
            A[2 * i, camera_indices * 6 + s] = 1
            A[2 * i + 1, camera_indices * 6 + s] = 1

        for s in range(3):
            A[2 * i, n_cameras * 6 + point_indices * 3 + s] = 1
            A[2 * i + 1, n_cameras * 6 + point_indices * 3 + s] = 1

        return A

    def _get_optimized_params(self, params, n_cameras, n_points):
        """ Parse optimization results to camera params and 3D points"""
        camera_params = params[: n_cameras * 6].reshape((n_cameras, 6))
        points_3d = params[n_cameras * 6 :].reshape((n_points, 3))
        return camera_params, points_3d

    def optimize(
        self, camera_params, points_3d, points_2d, camera_indices, point_indices
    ):
        """Apply bundle adjustment optimization

        Parameters
        ----------
        camera_params: array, shape (n_cameras, 6)
            Contains initial camera parameter estimates. For each row,
            the parameters must be organized as 3 rotations, then 3
            translations in the camera reference frame.

        points_3d: array, shape (n_points, 3)
            Contains inital 3D points coordinates estimates in the same
            reference frame.

        points_2d: array, shape (n_observations, 2)
            Contains the 2D coordinates of every observed 3D point projected
            using one of the cameras.

        camera_indices: array, shape (n_observations,)
            Contains camera indices for each observed projection. The i-th
            element of this array corresponds to the camera that generated
            the i-th 2D point.

        point_indices: array, shape (n_observations,)
            Contains 3D points indices for each observed projection. The i-th
            element of this array corresponds to the 3D point that generated
            the i-th 2D point.

        Returns
        -------
        optimized_cameras: array, shape (n_cameras, 6)
            Contains optimized camera parameters. Same order as input.

        optimized_points: array, shape (n_points, 3)
            Contains optimized 3D points coordinates.
        """

        # Get parameters of interest
        n_cameras = camera_params.shape[0]
        n_points = points_3d.shape[0]

        # Initilize optimization with
        x0 = np.hstack((camera_params.ravel(), points_3d.ravel()))

        # Build sparse matrix and run optimization
        A = self._bundle_adjustment_sparsity(
            n_cameras, n_points, camera_indices, point_indices
        )
        optim_res = least_squares(
            self._objective_function,
            x0,
            jac_sparsity=A,
            verbose=self.verbose,
            x_scale="jac",
            ftol=self.tol,
            method=self.method,
            args=(n_cameras, n_points, camera_indices, point_indices, points_2d),
        )

        # Return optimized params
        self.optimized_cameras, self.optimized_points = self._get_optimized_params(
            optim_res.x, n_cameras, n_points
        )

        return self.optimized_cameras, self.optimized_points

"""
Structure from Motion bundle adjusment module.

Based on: https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html

Authors: Caio Cancian, Felipe Moran

Version: 1.0.0

"""
from functools import partial

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from pipeline.config import BundleAdjustmentConfig
import cv2
from pipeline import utils


def _prepare_optimization_input(cloud, Rs, Ts, tracks, masks):
    assert len(Rs) == len(Ts) == len(tracks)

    camera_params = []
    points_2d = np.empty((0, 2), dtype=np.float_)
    camera_indexes = np.empty((0,), dtype=int)
    point_indexes = np.empty((0,), dtype=int)

    for R, T in zip(Rs, Ts):
        R, T = utils.invert_reference_frame(R, T)
        camera_params += [np.vstack((cv2.Rodrigues(R)[0], T)).reshape(-1)]
    camera_params = np.array(camera_params, dtype=np.float_)
    assert camera_params.shape == (len(Rs), 6)

    cloud_mask = utils.get_not_nan_index_mask(cloud)
    cloud_reindex = np.full(cloud.shape[0], None, dtype=np.float_)
    cloud_reindex[cloud_mask] = np.arange(len(cloud_mask))

    for index, (track, track_mask) in enumerate(zip(tracks, masks)):
        intersection_mask = utils.get_intersection_mask(cloud_mask, track_mask)
        # track_bool_mask = [item in intersection_mask for item in track_mask]
        track_bool_mask = np.isin(track_mask, intersection_mask)

        camera_indexes_row = np.full(len(intersection_mask), index)
        camera_indexes = np.append(camera_indexes, camera_indexes_row)

        point_indexes_row = cloud_reindex[intersection_mask].astype(int)
        point_indexes = np.append(point_indexes, point_indexes_row)

        points_2d_row = track[track_bool_mask]
        points_2d = np.vstack((points_2d, points_2d_row))

        assert (
            len(camera_indexes_row)
            == len(point_indexes_row)
            == len(points_2d_row)
        )

    assert len(camera_indexes) == len(point_indexes) == len(points_2d)

    points_3d = cloud[cloud_mask]

    return (
        camera_params,
        points_3d,
        points_2d,
        camera_indexes,
        point_indexes,
    )


def _parse_optimization_result(
    point_cloud, optimized_cameras, optimized_points
):
    # Convert back to pipeline style from BA style
    Rs = []
    Ts = []

    not_nan_mask = ~utils.get_nan_bool_mask(point_cloud)
    point_cloud[not_nan_mask] = optimized_points

    for camera in optimized_cameras:
        R = cv2.Rodrigues(camera[:3])[0]
        T = camera[3:].reshape((3, -1))
        R, T = utils.invert_reference_frame(R, T)
        Rs += [R]
        Ts += [T]

    return Rs, Ts, point_cloud


def _rotate(points, rot_vecs):
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


def _project(config, points, camera_params):
    """Convert 3-D points to 2-D by projecting onto images.

    No radial distortion is considered.
    """

    # Rotate and translate
    points_proj = _rotate(points, camera_params[:, :3])
    points_proj += camera_params[:, 3:6]

    # Divide by scale
    points_proj = points_proj[:, :2] / points_proj[:, 2, np.newaxis]

    # Set camera calibration params
    fx = config.camera_matrix[0, 0]
    cx = config.camera_matrix[0, 2]
    fy = config.camera_matrix[1, 1]
    cy = config.camera_matrix[1, 2]

    # Apply projection formula WITHOUT radial distortion
    points_proj[:, 0] = points_proj[:, 0] * fx + cx
    points_proj[:, 1] = points_proj[:, 1] * fy + cy

    return points_proj


def _objective_function(
    config,
    params,
    n_cameras,
    n_points,
    camera_indices,
    point_indices,
    points_2d,
):
    """Compute residuals.

    `params` contains camera parameters and 3-D coordinates.
    """
    camera_params = params[: n_cameras * 6].reshape((n_cameras, 6))
    points_3d = params[n_cameras * 6 :].reshape((n_points, 3))
    points_proj = _project(
        config, points_3d[point_indices], camera_params[camera_indices]
    )
    return (points_proj - points_2d).ravel()


def _bundle_adjustment_sparsity(
    n_cameras, n_points, camera_indices, point_indices
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


def _get_optimized_params(params, n_cameras, n_points):
    """ Parse optimization results to camera params and 3D points"""
    camera_params = params[: n_cameras * 6].reshape((n_cameras, 6))
    points_3d = params[n_cameras * 6 :].reshape((n_points, 3))
    return camera_params, points_3d


def optimize(
    config, camera_params, points_3d, points_2d, camera_indices, point_indices
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
    A = _bundle_adjustment_sparsity(
        n_cameras, n_points, camera_indices, point_indices
    )

    objective_function = partial(_objective_function, config)

    optim_res = least_squares(
        objective_function,
        x0,
        jac_sparsity=A,
        verbose=config.verbose,
        x_scale="jac",
        ftol=config.tol,
        method=config.method,
        args=(n_cameras, n_points, camera_indices, point_indices, points_2d,),
    )

    # Return optimized params
    (optimized_cameras, optimized_points,) = _get_optimized_params(
        optim_res.x, n_cameras, n_points
    )

    return optimized_cameras, optimized_points


def run(config, Rs, Ts, cloud, tracks, masks):
    (
        camera_params,
        points_3d,
        points_2d,
        camera_indexes,
        point_indexes,
    ) = _prepare_optimization_input(cloud, Rs, Ts, tracks, masks)

    # Optimize
    optimized_cameras, optimized_points = optimize(
        config=config,
        camera_params=camera_params,
        points_3d=points_3d,
        points_2d=points_2d,
        camera_indices=camera_indexes,
        point_indices=point_indexes,
    )
    Rs, Ts, cloud = _parse_optimization_result(
        point_cloud=cloud,
        optimized_cameras=optimized_cameras,
        optimized_points=optimized_points,
    )

    return Rs, Ts, cloud

import numpy as np

from pipeline import utils
from pipeline import bundle_adjustment
from pipeline.reconstruction_algorithms import (
    five_pt,
    triangulate,
    calculate_projection_error,
)
from pipeline.utils import visualize


def five_pt_init(config, tracks, masks):
    """
    Initialization sequence using 5 point algorithm.

    Returns None, None, None when only one track is supplied and Rs, Ts and cloud otherwise.
    If there are more then two tracks, the last two are conserved and used, the others are discarded.

    :param config: config object. See config.py for more information
    :param tracks: list of 2D feature vectors. Each vector has the shape Dx2
:param masks: list of index masks for each feature vector. Indexes refer to the position of the item in the cloud
    :return: tuple containine:
        :param Rs: list of R matrices
        :param Ts: list of T vectors
        :param cloud: point cloud with N points as a ndarray with shape Nx3
    """

    if len(tracks) == 1:
        return [None] * 3

    if len(tracks) > 2:
        tracks, masks = tracks[-2:], masks[-2:]

    # FRAME 0
    R, T = utils.init_rt()
    Rs, Ts = [R], [T]

    # FRAME 1
    R, T, points, indexes = five_pt(
        config.five_pt_algorithm, tracks, masks, R, T
    )

    if R is None:
        return [None] * 3

    points, indexes = triangulate(
        config.camera_matrix,
        R_1=Rs[-1].transpose(),
        T_1=np.matmul(Rs[-1].transpose(), -Ts[-1]),
        R_2=R.transpose(),
        T_2=np.matmul(R.transpose(), -T),
        tracks=tracks,
        masks=masks,
    )

    cloud = utils.points_to_cloud(points=points, indexes=indexes)

    Rs += [R]
    Ts += [T]

    return Rs, Ts, cloud

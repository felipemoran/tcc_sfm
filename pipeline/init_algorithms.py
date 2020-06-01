import numpy as np

from pipeline import utils
from pipeline import bundle_adjustment
from pipeline.reconstruction_algorithms import (
    five_pt,
    triangulate,
)


def five_pt_init(config, Rs, Ts, tracks, masks):
    # TODO: convert mask type

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

    R, T, points, index_mask = five_pt(
        config.five_pt_algorithm, tracks, masks, Rs[-1], Ts[-1]
    )

    points, index_mask = triangulate(
        config.five_pt_algorithm.camera_matrix,
        R_1=Rs[-1].transpose(),
        T_1=np.matmul(Rs[-1].transpose(), -Ts[-1]),
        R_2=R.transpose(),
        T_2=np.matmul(R.transpose(), -T),
        tracks=tracks,
        masks=masks,
    )

    cloud = utils.points_to_cloud(points=points, indexes=index_mask)

    Rs += [R]
    Ts += [T]

    if config.use_bundle_adjustment:
        Rs, Ts, cloud = bundle_adjustment.run(
            config.bundle_adjustment, Rs, Ts, cloud, tracks, masks
        )

    return Rs, Ts, cloud, tracks, masks


def three_frame_init(self, Rs, Ts, cloud, tracks):
    raise NotImplementedError

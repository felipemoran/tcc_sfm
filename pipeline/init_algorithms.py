import numpy as np

from pipeline import utils
from pipeline import bundle_adjustment
from pipeline.reconstruction_algorithms import (
    five_pt,
    triangulate,
    calculate_projection_errors,
)
from pipeline.utils import visualize


def five_pt_init(config, tracks, masks):
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


# def three_frame_init(config, tracks, masks):
#     assert len(tracks) > 0
#     cloud = None
#
#     required_frames = 4 if config.use_bundle_adjustment else 3
#
#     if len(tracks) < required_frames:
#         return None, None, None, tracks, masks, float("inf")
#         # R, T = utils.init_rt()
#         # return [R], [T], cloud, tracks, masks
#
#     elif len(tracks) > required_frames:
#         tracks, masks = tracks[-required_frames:], masks[-required_frames:]
#
#     # Frame 0: init ============================================================
#     R, T = utils.init_rt()
#     Rs, Ts = [R], [T]
#
#     # Frame 1: 5 pt algorithm + triangulate points =============================
#     R, T, points, index_mask = five_pt(
#         config.five_pt_algorithm, tracks, masks, Rs[-1], Ts[-1]
#     )
#     Rs += [R]
#     Ts += [T]
#
#     points, index_mask = triangulate(
#         config.five_pt_algorithm.camera_matrix,
#         R_1=Rs[-2].transpose(),
#         T_1=np.matmul(Rs[-2].transpose(), -Ts[-2]),
#         R_2=Rs[-1].transpose(),
#         T_2=np.matmul(Rs[-1].transpose(), -Ts[-1]),
#         tracks=tracks,
#         masks=masks,
#     )
#
#     # convert it to point cloud
#     cloud = utils.points_to_cloud(points=points, indexes=index_mask)
#
#     # Frame 2: solve pnp =======================================================
#     # calculate R, T with solve EPnP for new frame
#     R, T = solve_epnp(config.solve_pnp, tracks[-1], masks[-1], cloud)
#
#     # refine R and T
#     R, T = solve_pnp_iterative(
#         config.solve_pnp, tracks[-1], masks[-1], cloud, R, T
#     )
#
#     Rs += [R]
#     Ts += [T]
#
#     # Bundle Adjustment ========================================================
#     # run a bundle adjustment step (if enabled)
#     if config.use_bundle_adjustment:
#         assert len(tracks) == 4
#         # Frame 3: solve pnp only if BA is enabled =============================
#         R, T = solve_epnp(config.solve_pnp, tracks[-1], masks[-1], cloud)
#
#         # refine R and T
#         R, T = solve_pnp_iterative(
#             config.solve_pnp, tracks[-1], masks[-1], cloud, R, T
#         )
#         Rs += [R]
#         Ts += [T]
#
#         error_wo_ba = calculate_projection_errors(
#             config.solve_pnp.camera_matrix,
#             [Rs[3]],
#             [Ts[3]],
#             cloud,
#             [tracks[3]],
#             [masks[3]],
#         )
#         # visualize(config.solve_pnp.camera_matrix, Rs, Ts, cloud)
#
#         print(f"-- error wo ba: {error_wo_ba}")
#
#         Rs, Ts, cloud = bundle_adjustment.run(
#             config.bundle_adjustment, Rs[:3], Ts[:3], cloud, tracks[:3], masks
#         )
#
#         # Frame 3: solve pnp only if BA is enabled =============================
#         R, T = solve_epnp(config.solve_pnp, tracks[-1], masks[-1], cloud)
#
#         # refine R and T
#         R, T = solve_pnp_iterative(
#             config.solve_pnp, tracks[-1], masks[-1], cloud, R, T
#         )
#         Rs += [R]
#         Ts += [T]
#     else:
#         assert len(tracks) == 3
#
#     # Calculate Error ======================================================
#     error = calculate_projection_errors(
#         config.solve_pnp.camera_matrix,
#         [Rs[-1]],
#         [Ts[-1]],
#         cloud,
#         [tracks[-1]],
#         [masks[-1]],
#     )
#     # visualize(config.solve_pnp.camera_matrix, Rs, Ts, cloud)
#
#     return Rs, Ts, cloud, tracks, masks, error

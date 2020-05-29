import numpy as np
import cv2

from pipeline import utils


def solvepnp(config, track_slice, track_mask, cloud, R=None, T=None):
    # TODO: convert mask type

    if R is not None and T is not None:
        use_extrinsic_gress = True
    else:
        use_extrinsic_gress = False

    assert not (
        (R is None) ^ (T is None)
    ), "Either both R and T are None or none of the two"

    # create new mask based on existing point cloud's and newly created track's
    cloud_mask = utils.get_not_nan_index_mask(cloud)
    intersection_mask = utils.get_intersection_mask(cloud_mask, track_mask)
    track_bool_mask = np.isin(track_mask, intersection_mask)

    if len(intersection_mask) < config.min_number_of_points:
        return R, T

    # go back to camera's reference frame
    R, T = utils.invert_reference_frame(R, T)

    return_value, R, T = cv2.solvePnP(
        objectPoints=cloud[intersection_mask],
        imagePoints=track_slice[track_bool_mask],
        cameraMatrix=config.camera_matrix,
        distCoeffs=None,
        rvec=None if R is None else cv2.Rodrigues(R)[0],
        tvec=T,
        useExtrinsicGuess=use_extrinsic_gress,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    # convert from camera coordinate base to global
    R = cv2.Rodrigues(R)[0].transpose()
    T = np.matmul(R, -T)

    print()

    return R, T


def five_pt(config, tracks, prev_R, prev_T):
    if len(tracks[0]) < config.min_number_of_points:
        return None, None, None, None

    # We have no 3D point info so we calculate based on the two cameras
    E, five_pt_mask = cv2.findEssentialMat(
        points1=tracks[0],
        points2=tracks[1],
        cameraMatrix=config.camera_matrix,
        method=cv2.RANSAC,
        prob=config.probability,
        threshold=config.threshold,
        mask=None,
    )

    # print(
    #     f"E: {sum(five_pt_mask.squeeze()):3}/{five_pt_mask.shape[0]:3}",
    #     end="\t",
    # )

    _, R, T, pose_mask, points_4d = cv2.recoverPose(
        E=E,
        points1=tracks[0],
        points2=tracks[1],
        cameraMatrix=config.camera_matrix,
        distanceThresh=config.distance_threshold,
        mask=five_pt_mask.copy(),
    )

    # print(f"P: {sum(pose_mask.squeeze()):3}/{pose_mask.shape[0]:3}")

    # filter out 3d_points and point_indexes according to mask
    final_mask = pose_mask.squeeze().astype(np.bool)

    points_3d = cv2.convertPointsFromHomogeneous(
        points_4d.transpose()
    ).squeeze()[final_mask]

    # Convert it back to first camera base system
    R, T = utils.invert_reference_frame(R, T)

    # Then convert it all to camera 0's reference system
    points_3d = utils.translate_points_to_base_frame(prev_R, prev_T, points_3d)
    R, T = utils.compose_rts(R, T, prev_R, prev_T)

    return R, T, points_3d, final_mask


def triangulate(camera_matrix, R_1, T_1, R_2, T_2, tracks):
    if any([R_1 is None, T_1 is None, R_2 is None, T_2 is None,]):
        return None

    assert (
        tracks.shape[0] == 2
    ), "Can't do reprojection with {} cameras, 2 are needed".format(
        tracks.shape[0]
    )

    P1 = np.matmul(camera_matrix, np.hstack((R_1, T_1)))
    P2 = np.matmul(camera_matrix, np.hstack((R_2, T_2)))

    points_4d = cv2.triangulatePoints(
        projMatr1=P1,
        projMatr2=P2,
        projPoints1=tracks[0].transpose(),
        projPoints2=tracks[1].transpose(),
    )
    points_3d = cv2.convertPointsFromHomogeneous(
        points_4d.transpose()
    ).squeeze()

    return points_3d


def calculate_projection(config, tracks, masks, prev_R, prev_T, cloud):
    track_pair, pair_mask = utils.get_last_track_pair(tracks, masks)

    if len(track_pair) == 1:
        return (*utils.init_rt(), None)

    assert len(track_pair) == 2

    R, T, points, index_mask = None, None, None, None

    if config.use_five_pt_algorithm:
        R, T, points, bool_mask = five_pt(
            config.five_pt_algorithm, track_pair, prev_R, prev_T
        )
        index_mask = pair_mask[bool_mask]

    if config.use_solve_pnp:
        # refine R and T based on previous point cloud
        # result is in camera 0's coordinate system
        R, T = solvepnp(config.solvepnp, tracks[-1], masks[-1], cloud, R, T)

    if config.use_reconstruct_tracks:
        points = triangulate(
            config.camera_matrix,
            R_1=prev_R.transpose(),
            T_1=np.matmul(prev_R.transpose(), -prev_T),
            R_2=R.transpose(),
            T_2=np.matmul(R.transpose(), -T),
            tracks=track_pair,
        )
        index_mask = pair_mask

    return R, T, points, index_mask


def calculate_projection_error(camera_matrix, Rs, Ts, cloud, tracks, masks):
    # TODO: convert mask type

    if cloud is None:
        return float("inf")

    cloud_mask = utils.get_not_nan_index_mask(cloud)
    error = 0

    for index, (R, T, original_track, track_mask) in enumerate(
        zip(Rs, Ts, tracks, masks)
    ):
        intersection_mask = utils.get_intersection_mask(cloud_mask, track_mask)
        # track_bool_mask = [item in intersection_mask for item in track_mask]
        track_bool_mask = np.isin(track_mask, intersection_mask)

        R_cam, T_cam = utils.invert_reference_frame(R, T)
        R_cam_vec = cv2.Rodrigues(R_cam)[0]

        projection_track = cv2.projectPoints(
            cloud[intersection_mask], R_cam_vec, T_cam, camera_matrix, None,
        )[0].squeeze()

        delta = original_track[track_bool_mask] - projection_track
        error += np.linalg.norm(delta, axis=1).mean()

    return error / (index + 1)

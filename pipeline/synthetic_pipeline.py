import os
import cv2
import glob
import time
import copy
import random
import argparse
import itertools
import collections
import numpy as np
import open3d as o3d
import networkx as nt
import matplotlib.pyplot as plt

from math import pi, sqrt

np.set_printoptions(3, suppress=True)

TYPE_CALIBRATION_MATRIX = 0
TYPE_CAMERA = 1
TYPE_POINT = 2


def write_to_viz_file(camera_matrix, Rs, ts, points):
    with open('out/viz_data.csv', 'w') as out_file:
        out_file.write('{}\n')

        def convert_and_save_line(line):
            line = [str(item) for item in line]
            out_file.write(','.join(line))
            out_file.write('\n')

        line_elements = [TYPE_CALIBRATION_MATRIX, 0] + list(camera_matrix.flatten())
        convert_and_save_line(line_elements)

        for index, (R, t) in enumerate(zip(Rs, ts)):
            line_elements = [TYPE_CAMERA, index] + list((R).flatten()) + list((t).flatten())
            convert_and_save_line(line_elements)

        for point_id, point in points.items():
            line_elements = [TYPE_POINT, point_id] + list(point['avg_point'].flatten())
            if 'color' in point:
                line_elements += list(point['point_color'].flatten())

            convert_and_save_line(line_elements)


def _get_relative_movement(tracks, feature_pack_id, points_3d, point_indexes, camera_matrix):
    reconstruction_distance_threshold = 2

    num_points = tracks.shape[1]

    assert len(tracks) == 2, 'Reconstruction from more than 2 views not yet implemented'

    # Remove all points that don't have correspondence between frames
    track_mask = [bool((tracks[:, point_id] > 0).all()) for point_id in range(num_points)]
    trimmed_tracks = tracks[:, track_mask]

    if trimmed_tracks.shape[1] <= 5:
        print("Not enough points to run 5-point algorithm. Aborting")
        # Abort!
        return [None] * 4

    if points_3d is None:
        # We have no 3D point info so we calculate based on the two cameras
        E, five_pt_mask = cv2.findEssentialMat(trimmed_tracks[0], trimmed_tracks[1], camera_matrix, cv2.RANSAC,
                                               threshold=0.1
                                               )

        rep = int(round(20 * sum(five_pt_mask.squeeze()) / five_pt_mask.shape[0]))
        print('Find E --> {:2} / {:2}  {:2}'.format(sum(five_pt_mask.squeeze()), five_pt_mask.shape[0],
                                                    '#' * rep + '_' * (20 - rep)))

        retval, R, t, pose_mask, points_4d = cv2.recoverPose(E=E,
                                                             points1=trimmed_tracks[0],
                                                             points2=trimmed_tracks[1],
                                                             cameraMatrix=camera_matrix,
                                                             distanceThresh=100,
                                                             mask=five_pt_mask.copy()
                                                             # mask=None
                                                             )

        rep = int(round(20 * sum(pose_mask.squeeze()) / pose_mask.shape[0]))
        print('P: {:2} / {:2}  {:2}'.format(sum(pose_mask.squeeze()), pose_mask.shape[0], '#' * rep + '_' * (20 - rep)))

        # Convert it back to first camera base system
        R = R.transpose()
        t = np.matmul(R, -t)

        # filter out 3d_points and point_indexes according to mask
        final_mask = pose_mask.squeeze().astype(np.bool)
        points_3d = cv2.convertPointsFromHomogeneous(points_4d.transpose()).squeeze()
        points_3d = points_3d[final_mask]

        point_indexes = np.array(list(range(num_points)))
        point_indexes = point_indexes[track_mask]
        point_indexes = point_indexes[final_mask]

    else:
        # We do have some info on the 3D points. Let's calculate the rotation of the second camera

        retval, R, t = cv2.solvePnP(points_3d, tracks[1, point_indexes], camera_matrix, None)
        R = cv2.Rodrigues(R)[0]
        R = R.transpose()
        t = np.matmul(R, -t)  # convert from vector to matrix
        #
        # Rx = cv2.Rodrigues(np.array([0, 0, pi/2]))[0]
        #

    assert len(points_3d) == len(point_indexes)

    return R, t, points_3d, point_indexes


def synthetic_pipeline():
    # camera arbitraria
    camera_matrix = np.array([
        [500, 0.0, 500],
        [0.0, 500, 500],
        [0.0, 0.0, 1.0],
    ], dtype=np.float_)

    # cubo de pontos centrados em (10, 5, 0) de tamanho 2x2x2
    points_3d = np.array(list(itertools.product([9, 10, 11], [4, 5, 6], [-1, 0, 1])), dtype=np.float_)
    # points_3d = np.array(list(itertools.product([8, 9, 10, 11, 12], [4, 5, 6], [0])) +
    #                      list(itertools.product([9, 10, 11], [4, 5], [1])) +
    #                      list(itertools.product([10], [4], [2]))
    #                      , dtype=np.float_)

    # matrizes de rotação para o posicionamento das cameras
    r1 = cv2.Rodrigues(np.array([-pi / 2, 0., 0.]))[0]
    r2 = cv2.Rodrigues(np.array([0, -pi / 4, 0]))[0]
    r3 = cv2.Rodrigues(np.array([0, -pi / 2, 0]))[0]

    # vetores de translação das câmeras na base global
    ts = np.array([
        [10, 0, 0],
        [25, 5, 0],
        [10, 12.5, 0],
        [5, 5, 0],
        [10, 5, 5],
    ], dtype=np.float_)

    # vetores de rotação das cameras na base global
    Rs = np.array([
        r1,
        np.matmul(r1, r3),
        np.matmul(r1, np.matmul(r3, r2)),
        np.matmul(r1, np.matmul(r3, np.matmul(r3, r3))),
        np.matmul(r1, r1),
    ])

    # para cada câmera calcule a projeção de cada ponto e imprima
    tracks = [None] * len(Rs)
    for index, (R, t) in enumerate(zip(Rs, ts)):
        # convert to the camera base, important!
        t_cam = np.matmul(R.transpose(), -t)
        R_cam = R.transpose()
        R_cam_vec = cv2.Rodrigues(R_cam)[0]

        tracks[index] = cv2.projectPoints(points_3d, R_cam_vec, t_cam, camera_matrix, None)[0].squeeze()

    tracks = np.array(tracks)

    comp_R = np.eye(3)
    comp_t = np.zeros((3, 1))
    Rs = [comp_R]
    ts = [comp_t]

    points_3d = None
    point_indexes = None
    points = {}

    for pair_index in range(len(tracks) - 1):

        rel_R, rel_t, ret_points_3d, ret_point_indexes = _get_relative_movement(
            tracks[pair_index:pair_index + 2],
            0,
            points_3d,
            point_indexes,
            camera_matrix
        )

        if ret_points_3d is not None:
            points_3d = ret_points_3d
            point_indexes = ret_point_indexes
            points = {index: {'avg_point': point} for index, point in enumerate(points_3d)}

        # store everything for later use
        Rs += [rel_R]
        ts += [rel_t]

    print("Rs: ")
    [print(i) for i in Rs]
    print()
    print("Ts: ")
    [print(i) for i in ts]

    write_to_viz_file(camera_matrix, Rs, ts, points)

    os.system(os.path.join(os.getcwd(), 'visualizer', 'cmake-build-debug', 'visualizer') + ' ' +
              os.path.join(os.getcwd(), 'out', 'viz_data.csv'))


if __name__ == '__main__':
    synthetic_pipeline()

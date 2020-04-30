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


from math import pi
def playground_1():
    # camera arbitraria
    camera_matrix = np.array([
        [200, 0.0, 1000],
        [0.0, 200, 1000],
        [0.0, 0.0, 1.0],
    ], dtype=np.float_)

    # cubo de pontos centrados em (10, 5, 0) de tamanho 2x2x2
    points_3d = np.array(list(itertools.product([9, 11], [4, 6], [-1, 1])), dtype=np.float_)

    # matrizes de rotação para o posicionamento das cameras
    r0_1 = cv2.Rodrigues(np.array([pi/2, 0., 0.]))[0]
    r1_3 = cv2.Rodrigues(np.array([pi/2, 0, 0]))[0]
    r1_4 = cv2.Rodrigues(np.array([0., 0, -3*pi/4]))[0]

    # vetores de translação das câmeras
    ts = np.array([
        [-10, 0, 0],
        [-5, 0, 15],
    ], dtype=np.float_)

    # vetores de rotação das cameras baseados na multiplicação e conversão das matrizes de rotação apropriadas
    R_vecs = np.array([
        cv2.Rodrigues(r0_1)[0],
        cv2.Rodrigues(np.matmul(r1_3, r1_4))[0],
    ])

    # para cada câmera calcule a projeção de cada ponto e imprima
    tracks = [None]*2
    for index, (R_vec, t) in enumerate(zip(R_vecs, ts)):
        tracks[index] = cv2.projectPoints(points_3d, R_vec, t, camera_matrix, None)[0].squeeze()
        for p3d, p2d in zip(points_3d, tracks[index]):
            print(p3d, p2d)
        print()

    tracks = np.array(tracks)

    def _get_relative_movement(tracks, feature_pack_id):
        reconstruction_distance_threshold = 2

        num_points = tracks.shape[1]
        point_indexes = np.array(range(num_points)) + 200 * feature_pack_id

        assert len(tracks) == 2, 'Reconstruction from more than 2 views not yet implemented'

        # Remove all points that don't have correspondence between frames
        track_mask = [bool((tracks[:, point_index] > 0).all()) for point_index in range(num_points)]
        trimmed_tracks = tracks[:, track_mask]
        point_indexes = point_indexes[track_mask]

        if trimmed_tracks.shape[1] <= 5:
            # Abort!
            return [None] * 4

        E, five_pt_mask = cv2.findEssentialMat(trimmed_tracks[0], trimmed_tracks[1], camera_matrix, cv2.RANSAC,
                                               # threshold=0.1
                                               )

        rep = int(round(20 * sum(five_pt_mask.squeeze()) / five_pt_mask.shape[0]))
        irep = 20 - rep
        print('Find E --> {:2} / {:2}  {:2}'.format(sum(five_pt_mask.squeeze()), five_pt_mask.shape[0],
                                                    '#' * rep + '_' * irep))

        retval, R, t, pose_mask, points_4d = cv2.recoverPose(E=E,
                                                             points1=trimmed_tracks[0],
                                                             points2=trimmed_tracks[1],
                                                             cameraMatrix=camera_matrix,
                                                             distanceThresh=2,
                                                             mask=five_pt_mask
                                                             # mask=None
                                                             )

        rep = int(round(20 * sum(pose_mask.squeeze()) / pose_mask.shape[0]))
        irep = 20 - rep
        print('Pose   --> {:2} / {:2}  {:2}'.format(sum(pose_mask.squeeze()), pose_mask.shape[0], '#' * rep + '_' * irep))

        # print("R: {}".format(R))
        # print("t: {}".format(t))
        # TODO: filter out 3d_points and point_indexes according to mask
        final_mask = pose_mask.squeeze().astype(np.bool)
        points_3d = cv2.convertPointsFromHomogeneous(points_4d.transpose()).squeeze()
        points_3d = points_3d[final_mask]
        point_indexes = point_indexes[final_mask]

        print(cv2.Rodrigues(R)[0])
        print()
        print(t)

        return R, t, points_3d, point_indexes

    _get_relative_movement(tracks, 0)


def playground_2():
    TYPE_CALIBRATION_MATRIX = 0
    TYPE_CAMERA = 1
    TYPE_POINT = 2
    # define TYPE_POINT 2
    # point_coordinates = np.array(list(itertools.product([8, 9, 10, 11, 12], [4, 5, 6], [0])) +
    #                              list(itertools.product([9, 10, 11], [4, 5, 6], [1])) +
    #                              list(itertools.product([10], [4, 5, 6], [2]))
    #                              )
    point_coordinates = np.array(list(itertools.product([9, 10, 11], [4, 5, 6], [-1, 0, 1])))
    points = {index:{'avg_point':point} for index, point in enumerate(point_coordinates)}

    # matrizes de rotação para o posicionamento das cameras
    r0_1 = cv2.Rodrigues(np.array([pi / 2, 0., 0.]))[0]
    r1_3 = cv2.Rodrigues(np.array([pi / 2, 0, 0]))[0]
    r1_4 = cv2.Rodrigues(np.array([0., 0, - pi / 2]))[0]

    # vetores de translação das câmeras
    ts = np.array([
        [-10, 0, 0],
        [-5, 0, 15],
    ], dtype=np.float_)

    # vetores de rotação das cameras baseados na multiplicação e conversão das matrizes de rotação apropriadas
    Rs = np.array([
        r0_1,
        np.matmul(r1_3, r1_4),
    ])

    camera_matrix = np.array([[765.16859169, 0., 379.11876567],
                              [0., 762.38664643, 497.22086655],
                              [0., 0., 1.]])

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
                t = np.matmul(np.matmul(R, -t), R.transpose())
                R = np.linalg.inv(R)

                line_elements = [TYPE_CAMERA, index] + list((R).flatten()) + list((t).flatten())
                convert_and_save_line(line_elements)

            for point_id, point in points.items():
                line_elements = [TYPE_POINT, point_id] + list(point['avg_point'].flatten())
                if 'color' in point:
                    line_elements += list(point['point_color'].flatten())

                convert_and_save_line(line_elements)

    write_to_viz_file(camera_matrix, Rs, ts, points)

    R = Rs[1]
    t = ts[1]

    os.system(os.path.join(os.getcwd(), 'visualizer', 'cmake-build-debug', 'visualizer') + ' ' +
              os.path.join(os.getcwd(), 'out', 'viz_data.csv'))

if __name__ == '__main__':
    playground_2()



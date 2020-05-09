import numpy as np
import itertools
import cv2
from math import pi, sqrt
import matplotlib.pyplot as plt


np.set_printoptions(3, suppress=True)


def playground_1():
    camera_matrix = np.array([
        [500, 0.0, 500],
        [0.0, 500, 500],
        [0.0, 0.0, 1.0],
    ], dtype=np.float_)

    # points_3d = np.array(list(itertools.product([9, 10, 11], [4, 5, 6], [-1, 0, 1])), dtype=np.float_)
    points_3d = np.array(list(itertools.product([9, 11], [4, 6], [-1, 1])), dtype=np.float_)
    # points_3d = np.array(list(itertools.product([8, 9, 10, 11, 12], [4, 5, 6], [0])) +
    #                      list(itertools.product([8, 9, 10], [4, 5], [1])) +
    #                      list(itertools.product([8], [4], [2]))
    #                      , dtype=np.float_)
    r1 = cv2.Rodrigues(np.array([-pi / 2, 0., 0.]))[0]
    r2 = cv2.Rodrigues(np.array([0, -pi / 2, 0]))[0]
    r3 = cv2.Rodrigues(np.array([0, -pi / 2, 0]))[0]

    # vetores de rotação das cameras na base global
    Rs = np.array([
        r1,
        np.matmul(r1, r2),
        np.matmul(r1, np.matmul(r3, r2)),
        np.matmul(r1, np.matmul(r3, np.matmul(r3, r3))),
        # np.matmul(r1, r1),
    ])

    # vetores de translação das câmeras na base global
    Ts = np.array([
        [10, 0, 0],
        [15, 5, 0],
        [10, 10, 0],
        [5, 5, 0],
        # [10, 5, 5],
    ], dtype=np.float_)

    # para cada câmera calcule a projeção de cada ponto e imprima
    tracks = []
    for index, (R, t) in enumerate(zip(Rs, Ts)):
        # convert to the camera base, important!
        t_cam = np.matmul(R.transpose(), -t)
        R_cam = R.transpose()
        R_cam_vec = cv2.Rodrigues(R_cam)[0]

        track_slice = cv2.projectPoints(points_3d, R_cam_vec, t_cam, camera_matrix, None)[0].squeeze()
        # tracks[index] = track_slice
        tracks += [track_slice]

    tracks = np.array(tracks)

    recovered_points = []
    for i in range(len(tracks) - 1):
        track_slice_pair = tracks[i:i + 2]

        P = []
        for j in range(2):
            R = Rs[i+j]
            T = Ts[i+j]

            T_cam = np.matmul(R.transpose(), -T)
            R_cam = R.transpose()

            T_cam = T_cam
            R_cam = R_cam

            P_ = np.hstack((R_cam, T_cam.reshape((3, 1))))

            P += [np.matmul(camera_matrix, P_)]

        points = cv2.triangulatePoints(
            P[0],
            P[1],
            track_slice_pair[0].transpose(),
            track_slice_pair[1].transpose(),
        )
        points = cv2.convertPointsFromHomogeneous(points.transpose()).squeeze()

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection="3d")
        # ax.scatter3D(points.transpose()[0], points.transpose()[1], points.transpose()[2])
        # plt.show()

        print(points, end='\n\n')

        recovered_points += [points]

    recovered_points = np.array(recovered_points)


if __name__ == '__main__':
    playground_1()
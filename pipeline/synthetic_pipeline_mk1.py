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
from pipeline import utils
from pipeline.base_pipeline import BasePipeline
from pipeline.video_pipeline_mk1 import VideoPipelineMK1

np.set_printoptions(3, suppress=True)

TYPE_CALIBRATION_MATRIX = 0
TYPE_CAMERA = 1
TYPE_POINT = 2


class SyntheticPipelineMK1(VideoPipelineMK1):
    def __init__(self):
        # camera arbitraria
        self.camera_matrix = np.array([
            [500, 0.0, 500],
            [0.0, 500, 500],
            [0.0, 0.0, 1.0],
        ], dtype=np.float_)

        self.dir = None

    def _get_video(self, file_path):
        return None, None

    def _process_next_frame(self, file):
        # Replaces the original function to generate synthetic data while maintaining everything else

        # points_3d = np.array(list(itertools.product([9, 10, 11], [4, 5, 6], [-1, 0, 1])), dtype=np.float_)
        # points_3d = np.array(list(itertools.product([9, 11], [4, 6], [-1, 1])), dtype=np.float_)
        points_3d = np.array(list(itertools.product([8, 9, 10, 11, 12], [4, 5, 6], [0])) +
                             list(itertools.product([8, 9, 10], [4, 5], [1])) +
                             list(itertools.product([8], [4], [2]))
                             , dtype=np.float_)

        # matrizes de rotação para o posicionamento das cameras
        r1 = cv2.Rodrigues(np.array([-pi / 2, 0., 0.]))[0]
        r2 = cv2.Rodrigues(np.array([0, -pi / 4, 0]))[0]
        r3 = cv2.Rodrigues(np.array([0, -pi / 2, 0]))[0]

        # vetores de translação das câmeras na base global
        Ts = np.array([
            [10, 0, 0],
            [15, 5, 0],
            [10, 10, 0],
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

        # mimic otiginal function variables
        is_new_feature_set = True
        feature_pack_id = 0
        frame_counter = 0

        # para cada câmera calcule a projeção de cada ponto e imprima
        tracks = [None] * len(Rs)
        for index, (R, t) in enumerate(zip(Rs, Ts)):
            # convert to the camera base, important!
            t_cam = np.matmul(R.transpose(), -t)
            R_cam = R.transpose()
            R_cam_vec = cv2.Rodrigues(R_cam)[0]

            track_slice = cv2.projectPoints(points_3d, R_cam_vec, t_cam, self.camera_matrix, None)[0].squeeze()
            # tracks[index] = track_slice
            yield frame_counter, feature_pack_id, track_slice, is_new_feature_set

            is_new_feature_set = False
            frame_counter += 1


if __name__ == '__main__':
    sp = SyntheticPipelineMK1()
    sp.run()

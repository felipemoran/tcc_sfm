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
from pipeline.synthetic_pipeline_mk1 import SyntheticPipelineMK1
from pipeline.video_pipeline_mk2 import VideoPipelineMK2

np.set_printoptions(3, suppress=True)

TYPE_CALIBRATION_MATRIX = 0
TYPE_CAMERA = 1
TYPE_POINT = 2


class SyntheticPipelineMK2(SyntheticPipelineMK1, VideoPipelineMK2):
    def __init__(self):
        SyntheticPipelineMK1.__init__(self)
        # camera arbitraria
        self.camera_matrix = np.array([
            [500, 0.0, 500],
            [0.0, 500, 500],
            [0.0, 0.0, 1.0],
        ], dtype=np.float_)
        self.dir = None

    def _process_next_frame(self, file):
        return SyntheticPipelineMK1._process_next_frame(self, None)

    def _get_synthetic_camera_translations(self):
        # vetores de translação das câmeras na base global
        Ts = np.array([
            [10, 0, 0],
            [20, 5, 0],
            [10, 12.5, 0],
            [5, 5, 0],
            [10, 5, 5],
        ], dtype=np.float_)
        return Ts

    def run(self):
        return VideoPipelineMK2.run(self)


if __name__ == '__main__':
    sp = SyntheticPipelineMK2()
    sp.run()

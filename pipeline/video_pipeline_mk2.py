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

from os import path
from pipeline import utils
from pipeline.base_pipeline import BasePipeline


class VideoPipelineMK2(BasePipeline):
    def __init__(self, dir, save_debug_visualization=False):
        self.dir = dir
        self.save_debug_visualization = save_debug_visualization

        # Number of frames to skip between used frames (eg. 2 means using frames 1, 4, 7 and dropping 2, 3, 5, 6)
        self.frames_to_skip = 0

        # Number of frames for initial estimation
        self.num_frames_initial_estimation = 10

        # Number of frames between baseline reset. Hack to avoid further frames having no or few features
        self.feature_reset_rate = 10000

        # params for ShiTomasi corner detection
        self.feature_params = {
            "maxCorners": 200,
            "qualityLevel": 0.5,
            "minDistance": 30,
            "blockSize": 10
        }

        # Parameters for lucas kanade optical flow
        self.lk_params = {
            "winSize": (15, 15),
            "maxLevel": 2,
            "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        }

        self.image_size = None

        self.camera_matrix = np.array([[765.16859169, 0., 379.11876567],
                                       [0., 762.38664643, 497.22086655],
                                       [0., 0., 1.]])

        self.recover_pose_reconstruction_distance_threshold = 50
        self.find_essential_mat_threshold = 3
        self.find_essential_mat_prob = 0.98

        self.debug_colors = np.random.randint(0, 255, (self.feature_params['maxCorners'], 3))

    def run(self):
        # Start by finding the images
        file, filename = self._get_video(self.dir)

        prev_frame_id = None
        prev_feature_pack_id = None
        prev_track_slice = None

        points_3d = None
        point_indexes = None

        Rs = [np.eye(3)]
        Ts = [np.zeros((3, 1))]
        stored_points = np.zeros((0, 3))

        # Loop through frames (using generators)
        for (next_frame_id, next_feature_pack_id, next_track_slice, is_new_feature_set) \
                in self._process_next_frame(file):
            if is_new_feature_set:
                prev_frame_id = next_frame_id
                prev_feature_pack_id = next_feature_pack_id
                prev_track_slice = next_track_slice

                assert points_3d is None, 'Resetting KLT features is not yet implemented'
                continue

            assert prev_feature_pack_id == next_feature_pack_id

            if points_3d is None:
                R, T, points_3d, point_indexes = self._get_pose_from_two_tracks(
                    np.array([prev_track_slice, next_track_slice])
                )
                if len(points_3d) < 10:
                    points_3d = None
                    continue

                stored_points = np.append(stored_points, points_3d, axis=0)
            else:
                R, T = self._get_pose_from_points_and_projection(next_track_slice, points_3d, point_indexes)

            Rs += [R]
            Ts += [T]

            # prepare everything for next round
            prev_frame_id = next_frame_id
            prev_feature_pack_id = next_feature_pack_id
            prev_track_slice = next_track_slice

        # when all frames are processed, plot result
        utils.write_to_viz_file(self.camera_matrix, Rs, Ts, stored_points)
        utils.call_viz()

    # =================== INTERNAL FUNCTIONS ===========================================================================


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dir',
                        help='Directory with image files for reconstructions')
    parser.add_argument('-sd', '--save_debug_visualization',
                        help='Save debug visualizations to files?', action='store_true', default=False)
    args = parser.parse_args()

    sfm = VideoPipelineMK2(**vars(args))
    sfm.run()

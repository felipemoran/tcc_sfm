import os
import cv2
import glob
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


class SimplePipeline:

    def __init__(self, dir):
        self.dir = dir

        # Number of frames to skip between used frames (eg. 2 means using frames 1, 4, 7 and dropping 2, 3, 5, 6)
        self.frames_to_skip = 10

        # Number of frames for initial estimation
        self.num_frames_initial_estimation = 10

        # Number of frames between baseline reset. Hack to avoid further frames having no or few features
        self.feature_reset_rate = 10

        # params for ShiTomasi corner detection
        self.feature_params = {
            "maxCorners": 200,
            "qualityLevel": 0.3,
            "minDistance": 7,
            "blockSize": 7
        }

        # Parameters for lucas kanade optical flow
        self.lk_params = {
            "winSize": (15, 15),
            "maxLevel": 2,
            "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        }

        self.image_size = None
        self.k = None
        self.k_ = None

    def run(self):
        # Start by finding the images
        file, filename = self._get_video(self.dir)

        prev_frame_id = None
        prev_feature_pack_id = None
        prev_track_slice = None

        # Loop through frames (using generators)
        for (next_frame_id, next_feature_pack_id, next_track_slice, is_new_feature_set) in self._process_next_frame(file):
            if is_new_feature_set:
                prev_frame_id = next_frame_id
                prev_feature_pack_id = next_feature_pack_id
                prev_track_slice = next_track_slice
                continue

            # TODO: calculate 2 camera positions
            # TODO: translate point and camera positions back to reference frame (camera 0 probably)
            # TODO: store this data somewhere. Keep in mind that a same point might have multiple calculated positions

        # TODO: when all frames are processed, plot result
        # TODO: but to do that, first average point positions when there are multiple

    # =================== INTERNAL FUNCTIONS ===========================================================================

    @staticmethod
    def _get_video(file_path):
        print('Looking for files in {}'.format(dir))

        assert path.isfile(file_path), 'Invalid file location'

        file = cv2.VideoCapture(file_path)
        filename = os.path.basename(file_path)

        return file, filename

    def _process_next_frame(self, file):
        reset_features = True
        next_frame = None

        # this counter is actually for processed frames and not for raw frames
        frame_counter = 0

        while True:
            if reset_features:
                reset_features = False
                ref_frame = frame_counter

                # check if we're in the first iteration
                if next_frame is None:
                    # if yes, get the first frame
                    ret, next_frame = file.read()
                    assert ret, 'File has no frames!'
                    next_frame = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

                # generate new feature set
                next_features = cv2.goodFeaturesToTrack(next_frame, mask=None, **self.feature_params)

                # reset control variables
                num_features = len(next_features)
                track_indexes = np.array(range(len(next_features)))
                is_new_feature_set = True

                # mimics the same format as status from cv2.calcOpticalFlowPyrLK()
                status = np.full((len(next_features), 1), True)

            else:
                # Read frame
                # skip some frames between frame reads. The last one is a useful frame
                for _ in range(self.frames_to_skip + 1):
                    ret, next_frame = file.read()
                    if not ret:
                        return

                frame_counter += 1

                next_frame = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

                # calculate optical flow
                next_features, status, err = cv2.calcOpticalFlowPyrLK(
                    prev_frame,
                    next_frame,
                    prev_features,
                    None,
                    **self.lk_params
                )

                if frame_counter - ref_frame == self.feature_reset_rate:
                    reset_features = True

            # generate track slice
            track_indexes, next_features, track_slice = self._features_to_track_slice(
                num_features=num_features,
                track_indexes=track_indexes,
                frame_features=next_features,
                status=status
                # mimics the same format as status from cv2.calcOpticalFlowPyrLK()
            )

            prev_frame = next_frame.copy()  # do I really need this .copy() ?
            prev_features = next_features

            yield frame_counter, ref_frame, track_slice, is_new_feature_set
            is_new_feature_set = False

    @staticmethod
    def _features_to_track_slice(num_features, track_indexes, frame_features, status):
        status = status.squeeze().astype(np.bool)

        # remove from track_indexes those indexes that are not valid anymore according to frame_features_status
        track_indexes = track_indexes[status]

        # remove features that are not valid
        frame_features = frame_features.squeeze()
        frame_features = frame_features[status]

        # create track slice
        track_slice = np.zeros(
            (
                2,  # x and y coordinates for a point in an image
                num_features,  # number of features detected on reference frame for track
            ), dtype=np.float_)

        # and populate it
        for feature_index, track_index in enumerate(track_indexes):
            track_slice[0][track_index] = frame_features[feature_index][0]
            track_slice[1][track_index] = frame_features[feature_index][1]

        return track_indexes, frame_features, track_slice


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dir',
                        help='Directory with image files for reconstructions')
    # parser.add_argument('-mrate', '--match_survival_rate', type=float,
    #                     help='Survival rate of matches to consider image pair success', default=0.5)
    # parser.add_argument('-viz', '--visualize',
    #                     help='Visualize the sparse point cloud reconstruction?', action='store_true', default=False)
    # parser.add_argument('-sd', '--save_debug_visualization',
    #                     help='Save debug visualizations to files?', action='store_true', default=False)
    # parser.add_argument('-mvs', '--save_mvs',
    #                     help='Save reconstruction to an .mvs file? Provide filename')
    # parser.add_argument('-sc', '--save_cloud',
    #                     help='Save reconstruction to a point cloud file (PLY, XYZ and OBJ). Provide filename')

    args = parser.parse_args()

    sfm = SimplePipeline(**vars(args))
    sfm.run()

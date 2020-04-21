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

        stored_cameras = []
        stored_points = {}

        # Loop through frames (using generators)
        for (next_frame_id, next_feature_pack_id, next_track_slice, is_new_feature_set) \
                in self._process_next_frame(file):
            if is_new_feature_set:
                prev_frame_id = next_frame_id
                prev_feature_pack_id = next_feature_pack_id
                prev_track_slice = next_track_slice
                continue

            assert prev_feature_pack_id == next_feature_pack_id

            tracks = np.array([prev_track_slice, next_track_slice], next_feature_pack_id)

            # TODO: calculate 2 camera positions
            rel_R, rel_t, points_3d, points_indexes = self._get_relative_movement(tracks)

            # TODO: translate camera position back to reference frame
            comp_R, comp_t = self._compose_movement(comp_R, comp_t, rel_R, rel_t)

            # TODO: translate 3D point positiona back to reference frame
            points_3d = self._translate_point_positions(comp_R, comp_t, points_3d)

            # TODO: store everything for later use
            stored_points = self._store_new_points(stored_points, points_3d, points_indexes)
            stored_cameras += [rel_R, rel_t]

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

        # create track slice (Nx2)
        track_slice = np.full(
            (
                num_features,  # number of features detected on reference frame for track
                2,  # x and y coordinates for a point in an image
            ), -1, dtype=np.float_)

        # and populate it
        for feature_index, track_index in enumerate(track_indexes):
            track_slice[track_index] = frame_features[feature_index]
            # track_slice[0][track_index] = frame_features[feature_index][0]
            # track_slice[1][track_index] = frame_features[feature_index][1]

        return track_indexes, frame_features, track_slice

    def _get_relative_movement(self, tracks, feature_pack_id):
        num_points = tracks.shape[1]
        point_indexes = np.array(range(num_points)) + self.feature_params['maxCorners']

        assert len(tracks) == 2, 'Reconstruction from more than 2 views not yet implemented'

        # Remove all points that don't have correspondence between frames
        mask = [bool((tracks[:, point_index] > 0).all()) for point_index in range(num_points)]
        trimmed_tracks = tracks[:, mask]
        point_indexes = point_indexes[mask]

        E, mask = cv2.findEssentialMat(trimmed_tracks[0], trimmed_tracks[1], self.camera_matrix, cv2.RANSAC)
        retval, R, t, mask, points_3d = cv2.recoverPose(E,
                                                        trimmed_tracks[0],
                                                        trimmed_tracks[1],
                                                        self.camera_matrix,
                                                        distanceThresh=self.reconstruction_distance_threshold,
                                                        mask=mask
                                                        )
        # TODO: test

        # TODO: filter out 3d_points and point_indexes according to mask

        return R, t, points_3d, point_indexes

    @staticmethod
    def _compose_movement(comp_R, comp_t, rel_R, rel_t):
        # use cv2.composeRT(rvec1, tvec1, rvec2, tvec2) and get first return value
        # TODO: implement
        return None, None

    @staticmethod
    def _translate_point_positions(comp_R, comp_t, points_3d):
        # TODO: implement
        return points_3d

    @staticmethod
    def _store_new_points(stored_points, points_3d, points_indexes):
        for point_index, point in zip(points_indexes, points_3d):
            if point_index not in stored_points:
                stored_points[point_index] = {'accum': point, 'count': 1}
                continue

            stored_points[point_index]['count'] += 1
            stored_points[point_index]['accum'] += point
            # TODO: check is this is valid

        return stored_points

    if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dir',
                        help='Directory with image files for reconstructions')

    args = parser.parse_args()

    sfm = SimplePipeline(**vars(args))
    sfm.run()

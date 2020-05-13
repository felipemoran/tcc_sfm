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


class StructureFromMotion:

    MATCH_RATIO_THRESHOLD = 0.8

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
            "blockSize": 7,
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

        # Then get first few frames for initial pose/point position estimation
        rs, ts, points_3d = self._get_initial_estimation(file)

        # Setup bundle adjuster
        bundle_adjuster = self._setup_bundle_adjuster(file, ts, rs, points_3d)

        # Then keep processing frames until end of file
        rs, ts, points_3d = self._process_remaining_frames(file, bundle_adjuster)

        # Finally, show result
        self._visualize_3d(rs, ts, points_3d)

    # =================== INTERNAL FUNCTIONS ===========================================================================

    @staticmethod
    def _get_video(file_path):
        print("Looking for files in {}".format(dir))

        assert path.isfile(file_path), "Invalid file location"

        file = cv2.VideoCapture(file_path)
        filename = os.path.basename(file_path)

        return file, filename

    def _get_initial_estimation(self, file):
        print("Getting initial estimation of camera pose and scene")
        print("Generating tracks")
        tracks = self._get_initial_feature_tracks(file)
        print("Estimating scene")
        rs, ts, points_3d = self._reconstruct_from_tracks(tracks)
        print("Scene estimated")
        return rs, ts, points_3d

    def _get_initial_feature_tracks(self, file):
        print("Processing frames: ", end="")

        # Get first frame and features
        frame_index = 0
        ret, prev_frame = file.read()
        assert ret, "Error reading first frame of file"
        prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        prev_features = cv2.goodFeaturesToTrack(
            prev_frame, mask=None, **self.feature_params
        )

        track_indexes = np.array(range(len(prev_features)))
        feature_tracks = np.zeros(
            (
                self.num_frames_initial_estimation,  # number of views
                2,  # x and y coordinates for a point in an image
                len(prev_features),  # number of feature tracks
            ),
            dtype=np.float_,
        )

        self._save_features_to_tracks(
            track_indexes,
            feature_tracks,
            frame_index,
            prev_features,
            np.full(
                (len(prev_features), 1), True
            ),  # mimics the same format as status from cv2.calcOpticalFlowPyrLK()
        )
        print("{}".format(frame_index), end="")

        while frame_index < self.num_frames_initial_estimation - 1:
            # skip some frames between frame reads. The last one is a useful frame
            for _ in range(self.frames_to_skip + 1):
                ret, next_frame = file.read()
                assert (
                    ret
                ), "Error reading frame during initial estimation. File seems to be too short"

            frame_index += 1
            print(", {}".format(frame_index), end="")

            # convert it to grayscale
            next_frame = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

            # calculate optical flow
            next_features, status, err = cv2.calcOpticalFlowPyrLK(
                prev_frame, next_frame, prev_features, None, **self.lk_params
            )

            # Save good features to their tracks
            track_indexes, next_features = self._save_features_to_tracks(
                track_indexes, feature_tracks, frame_index, next_features, status
            )

            prev_frame = next_frame.copy()  # do I really need this .copy() ?
            prev_features = next_features

        self.image_size = next_frame.shape
        print(" DONE")

        return feature_tracks

    def _save_features_to_tracks(
        self,
        track_indexes,
        feature_tracks,
        frame_index,
        frame_features,
        frame_features_status,
    ):
        frame_features_status = frame_features_status.squeeze().astype(np.bool)

        # remove from track_indexes those indexes that are not valid anymore according to frame_features_status
        track_indexes = track_indexes[frame_features_status]

        # remove features that are not valid
        frame_features = frame_features.squeeze()
        frame_features = frame_features[frame_features_status]

        for feature_index, track_index in enumerate(track_indexes):
            feature_tracks[frame_index][0][track_index] = frame_features[feature_index][
                0
            ]
            feature_tracks[frame_index][1][track_index] = frame_features[feature_index][
                1
            ]

        return track_indexes, frame_features

    def _reconstruct_from_tracks(self, tracks):
        print("Reconstructing from {} tracks".format(np.shape(tracks)[2]))

        f = max(self.image_size)
        if self.k is None:
            self.k = np.array(
                [
                    [f, 0, self.image_size[0] / 2],
                    [0, f, self.image_size[1] / 2],
                    [0, 0, 1],
                ]
            )

        (rs, ts, k_, points_3d) = cv2.sfm.reconstruct(
            points2d=tracks,
            K=self.k,
            Rs=None,
            Ts=None,
            points3d=None,
            is_projective=True,
        )
        self.k_ = k_

        print(
            "Estimated 3D points: {} / {}".format(len(points_3d), np.shape(tracks)[2])
        )
        print("Estimated views: {} / {}".format(len(rs), tracks.shape[0]))

        # if len(rs) != tracks.shape[0]:
        #     print("Unable to reconstruct all camera views ({}/{})".format(tracks.shape[0], len(rs)))
        #     # return
        #
        # if np.shape(tracks)[2] > len(points_3d):
        #     print("Unable to reconstruct all tracks ({}/{})".format(len(points_3d), np.shape(tracks)[2]))

        print("Refined intrinsics: ")
        print(k_)

        # points_3d_colors = np.zeros([np.shape(points_3d)[0], 3])
        # for index, point in enumerate(points_3d):
        #     for file, view_r, view_t in zip(files, rs, ts):
        #         point_2d, _ = cv2.projectPoints(point, view_r, view_t, k_, distCoeffs=None)
        #         point_2d = point_2d[0][0]
        #         point_2d = point_2d.astype(int)
        #
        #         if 0 <= point_2d[0] < np.shape(file)[0] and \
        #                 0 <= point_2d[1] < np.shape(file)[1]:
        #             points_3d_colors[index] = file[point_2d[0]][point_2d[1]]
        #             break

        return rs, ts, points_3d

    def _setup_bundle_adjuster(self, file, ts, rs, points_3d):
        # TODO: implement
        bundle_adjuster = None
        return bundle_adjuster

    def _process_remaining_frames(self, file):
        print("Extracting features")

        frame_counter = 0
        ret, prev_frame = file.read()
        assert not ret

        # TODO: finish implementing
        while True:
            if frame_counter % self.feature_reset_rate == 0:
                prev_features = cv2.goodFeaturesToTrack(
                    prev_frame, mask=None, **self.feature_params
                )

            ret, next_frame = file.read()
            if not ret:
                break

            next_frame = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

            # calculate optical flow
            next_features, status, err = cv2.calcOpticalFlowPyrLK(
                prev_frame, next_frame, prev_features, None, **self.lk_params
            )

            frame_counter += 1

    def _visualize_3d(self, rs, ts, points_3d, points_3d_colors):

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.squeeze(points_3d))
        o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", help="Directory with image files for reconstructions")
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

    sfm = StructureFromMotion(**vars(args))
    sfm.run()

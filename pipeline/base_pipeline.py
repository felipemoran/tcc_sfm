import cv2
import os
import time
from os import path
import numpy as np

from abc import ABC, abstractmethod
from pipeline import utils


class BasePipeline:
    def run(self):
        assert False, "This method needs to be implemented"

    @staticmethod
    def _get_video(file_path):
        print("Looking for files in {}".format(dir))

        assert path.isfile(file_path), "Invalid file location"

        file = cv2.VideoCapture(file_path)
        filename = os.path.basename(file_path)

        return file, filename

    def _process_next_frame(self, file):
        reset_features = True
        next_frame = None

        mask = None

        # this counter is actually for processed frames and not for raw frames
        frame_counter = 0

        while True:
            if reset_features:
                reset_features = False
                ref_frame = frame_counter

                # check if we're in the first iteration
                if next_frame is None:
                    # if yes, get the first frame
                    ret, next_frame_color = file.read()
                    assert ret, "File has no frames!"
                    next_frame = cv2.cvtColor(next_frame_color, cv2.COLOR_BGR2GRAY)

                # generate new feature set
                next_features = cv2.goodFeaturesToTrack(
                    next_frame, mask=None, **self.feature_params
                )

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
                    ret, next_frame_color = file.read()
                    if not ret:
                        return

                frame_counter += 1
                print("Frame {}: ".format(frame_counter), end="")

                next_frame = cv2.cvtColor(next_frame_color, cv2.COLOR_BGR2GRAY)

                # calculate optical flow
                next_features, status, err = cv2.calcOpticalFlowPyrLK(
                    prev_frame, next_frame, prev_features, None, **self.lk_params
                )

                if frame_counter - ref_frame == self.feature_reset_rate:
                    reset_features = True

                if self.save_debug_visualization:
                    self._display_klt_debug_frame(
                        next_frame_color,
                        status,
                        next_features,
                        prev_features,
                        track_indexes,
                    )

            # generate track slice
            track_indexes, next_features, track_slice = self._features_to_track_slice(
                num_features=num_features,
                track_indexes=track_indexes,
                frame_features=next_features,
                status=status
                # mimics the same format as status from cv2.calcOpticalFlowPyrLK()
            )

            yield track_slice, track_indexes, is_new_feature_set

            prev_frame = next_frame.copy()  # do I really need this .copy() ?
            prev_features = next_features

            is_new_feature_set = False

    def _display_klt_debug_frame(
        self, next_frame_color, status, next_features, prev_features, track_indexes
    ):
        # if mask is None:
        #     mask = np.zeros_like(next_frame_color)
        mask = np.zeros_like(next_frame_color)

        for feature_index, feature_status in enumerate(status):
            if feature_status[0] == 0:
                continue
            next_x, next_y = next_features[feature_index]
            prev_x, prev_y = prev_features[feature_index]

            mask = cv2.line(
                mask,
                (next_x, next_y),
                (prev_x, prev_y),
                self.debug_colors[feature_index].tolist(),
                2,
            )
            next_frame_color = cv2.circle(
                next_frame_color,
                (next_x, next_y),
                5,
                self.debug_colors[track_indexes[feature_index]].tolist(),
                -1,
            )

        img = cv2.add(next_frame_color, mask)

        cv2.imshow("frame", img)
        time.sleep(0.2)

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
            ),
            None,
            dtype=np.float_,
        )

        # and populate it
        for feature_index, track_index in enumerate(track_indexes):
            track_slice[track_index] = frame_features[feature_index]

        return track_indexes, frame_features, track_slice

    def _get_pose_from_points_and_projection(
        self, track_slice, points_3d, R=None, T=None
    ):
        # TODO: try with useExtrinsicGuess=True
        if R is not None and T is not None:
            use_extrinsic_gress = True
        else:
            use_extrinsic_gress = False

        assert not (
            (R is None) ^ (T is None)
        ), "Either both R and T are None or none of the two"

        retval, R, T = cv2.solvePnP(
            objectPoints=points_3d,
            imagePoints=track_slice,
            cameraMatrix=self.camera_matrix,
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

    def _get_pose_from_two_tracks(self, tracks):
        # Remove all points that don't have correspondence between frames
        num_points = tracks.shape[1]
        track_mask = [
            bool((tracks[:, point_id] > 0).all()) for point_id in range(num_points)
        ]
        tracks = tracks[:, track_mask]

        if tracks.shape[1] <= 5:
            print("Not enough points to run 5-point algorithm. Aborting")
            return None, None, [], []

        # We have no 3D point info so we calculate based on the two cameras
        E, five_pt_mask = cv2.findEssentialMat(
            ponts1=tracks[0],
            points2=tracks[1],
            cameraMatrix=self.camera_matrix,
            method=cv2.RANSAC,
            threshold=self.find_essential_mat_threshold,
            prob=self.find_essential_mat_prob,
        )

        print(
            "P: {}".format(
                utils.progress_bar(sum(five_pt_mask.squeeze()), five_pt_mask.shape[0])
            ),
            end="   ",
        )
        result = cv2.recoverPose(
            E=E,
            points1=tracks[0],
            points2=tracks[1],
            cameraMatrix=self.camera_matrix,
            distanceThresh=self.recover_pose_reconstruction_distance_threshold,
            mask=five_pt_mask.copy()
            # mask=None
        )
        _, R, T, pose_mask, points_4d = result

        print(
            "P: {}".format(
                utils.progress_bar(sum(pose_mask.squeeze()), pose_mask.shape[0])
            )
        )

        # Convert it back to first camera base system
        R, T = R.transpose(), np.matmul(R.transpose(), -T)

        # filter out 3d_points and point_indexes according to mask
        final_mask = pose_mask.squeeze().astype(np.bool)
        points_3d = cv2.convertPointsFromHomogeneous(points_4d.transpose()).squeeze()
        points_3d = points_3d[final_mask]

        # create point index mask
        point_indexes = np.arange(num_points)[track_mask][final_mask]

        assert len(points_3d) == len(point_indexes)

        return R, T, points_3d, point_indexes

    def _reproject_tracks_to_3d(self, R_1, T_1, R_2, T_2, tracks):
        assert (
            tracks.shape[0] == 2
        ), "Can't do reprojection with {} cameras, 2 are needed".format(tracks.shape[0])

        # Remove all points that don't have correspondence between frames
        num_points = tracks.shape[1]
        track_mask = [
            bool((tracks[:, point_id] > 0).all()) for point_id in range(num_points)
        ]
        points_index_mask = np.arange(num_points)[track_mask]
        tracks = tracks[:, track_mask]

        P1 = np.matmul(self.camera_matrix, np.hstack((R_1, T_1)))
        P2 = np.matmul(self.camera_matrix, np.hstack((R_2, T_2)))

        points_4d = cv2.triangulatePoints(
            projMatr1=P1,
            projMatr2=P2,
            projPoints1=tracks[0].transpose(),
            projPoints2=tracks[1].transpose(),
        )
        points_3d = cv2.convertPointsFromHomogeneous(points_4d.transpose()).squeeze()

        return points_3d, points_index_mask

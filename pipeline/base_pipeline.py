import cv2
import os
import time
from os import path
import numpy as np

from pipeline import utils


debug_colors = np.random.randint(0, 255, (200, 3))


class BasePipeline:
    def run(self, dir):
        assert False, "This method needs to be implemented"

    @staticmethod
    def _get_video(file_path):
        print("Looking for files in {}".format(dir))

        assert path.isfile(file_path), "Invalid file location"

        file = cv2.VideoCapture(file_path)
        filename = os.path.basename(file_path)

        return file, filename

    def _frame_skipper(self, file, frames_to_skip):
        while True:
            for _ in range(frames_to_skip + 1):
                ret, color_frame = file.read()
                if not ret:
                    return
            bw_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)
            yield color_frame, bw_frame

    def _file_to_tracks(self, file):
        config = self.config.klt
        reset_features = True

        prev_frame = None
        prev_features = np.empty((0, 2), dtype=np.float_)
        features = np.empty((0, 2), dtype=np.float_)
        indexes = np.empty((0,))
        start_index = 100

        for counter, (color_frame, bw_frame) in enumerate(
            self._frame_skipper(file, config.frames_to_skip)
        ):
            if prev_frame is not None:
                features, status, err = cv2.calcOpticalFlowPyrLK(
                    prevImg=prev_frame,
                    nextImg=bw_frame,
                    prevPts=prev_features,
                    nextPts=None,
                    winSize=(
                        config.optical_flow.window_size.height,
                        config.optical_flow.window_size.width,
                    ),
                    maxLevel=config.optical_flow.max_level,
                    criteria=(
                        config.optical_flow.criteria.criteria_sum,
                        config.optical_flow.criteria.max_iter,
                        config.optical_flow.criteria.eps,
                    ),
                )
                status = status.squeeze().astype(np.bool)
                indexes = indexes[status]
                features = features.squeeze()[status]

            if reset_features or counter % config.reset_period == 0:
                reset_features = False

                new_features = cv2.goodFeaturesToTrack(
                    image=bw_frame,
                    maxCorners=config.corner_selection.max_corners,
                    qualityLevel=config.corner_selection.quality_level,
                    minDistance=config.corner_selection.min_distance,
                    mask=None,
                    blockSize=config.corner_selection.block_size,
                ).squeeze()

                features, indexes = self._match_features(
                    features, indexes, new_features, start_index
                )
                start_index = max(indexes) + 1

            yield features, indexes
            prev_frame, prev_features = bw_frame, features

            if self.display_klt_debug_frames:
                self._display_klt_debug_frame(
                    color_frame, features, prev_features, indexes,
                )

    def _display_klt_debug_frame(
        self, color_frame, features, prev_features, indexes,
    ):
        mask = np.zeros_like(color_frame)

        for feature, prev_feature, index in zip(
            features, prev_features, indexes
        ):
            # next_x, next_y = feature
            # prev_x, prev_y = prev_feature

            mask = cv2.line(
                mask,
                # (next_x, next_y),
                # (prev_x, prev_y),
                tuple(feature),
                tuple(prev_feature),
                debug_colors[index].tolist(),
                2,
            )
            color_frame = cv2.circle(
                color_frame,
                tuple(feature),
                5,
                debug_colors[index].tolist(),
                -1,
            )

        img = cv2.add(color_frame, mask)

        cv2.imshow("frame", img)
        cv2.waitKey(200)

    def _match_features(
        self, old_features, old_indexes, new_features, index_start
    ):
        if len(old_features) == 0:
            return new_features, np.arange(len(new_features)) + index_start

        # TODO: add to config
        closeness_threshold = 10

        old_repeated = np.repeat(
            old_features, new_features.shape[0], axis=0
        ).reshape((old_features.shape[0], new_features.shape[0], 2))

        new_repeated = (
            np.repeat(new_features, old_features.shape[0], axis=0)
            .reshape((new_features.shape[0], old_features.shape[0], 2))
            .transpose((1, 0, 2))
        )

        distances = np.linalg.norm(old_repeated - new_repeated, axis=2)
        close_points = distances < closeness_threshold

        new_points_mask = np.arange(len(new_features))[
            close_points.sum(axis=0) == 0
        ]
        new_points = new_features[new_points_mask]
        new_indexes = np.arange(len(new_points)) + index_start

        # TODO: limit number of returned features

        features = np.vstack((old_features, new_features[new_points_mask]))
        indexes = np.concatenate((old_indexes, new_indexes))

        assert len(features) == len(indexes)

        return features, indexes

    def _run_solvepnp(self, track_slice, track_mask, cloud, R=None, T=None):
        # TODO: convert mask type

        config = self.config.solvepnp

        if R is not None and T is not None:
            use_extrinsic_gress = True
        else:
            use_extrinsic_gress = False

        assert not (
            (R is None) ^ (T is None)
        ), "Either both R and T are None or none of the two"

        # create new mask based on existing point cloud's and newly created track's
        cloud_mask = utils.get_not_nan_index_mask(cloud)
        intersection_mask = utils.get_intersection_mask(cloud_mask, track_mask)
        track_bool_mask = np.isin(track_mask, intersection_mask)

        if len(intersection_mask) < config.min_number_of_points:
            return R, T

        # go back to camera's reference frame
        R, T = utils.invert_reference_frame(R, T)

        return_value, R, T = cv2.solvePnP(
            objectPoints=cloud[intersection_mask],
            imagePoints=track_slice[track_bool_mask],
            cameraMatrix=self.config.camera_matrix,
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

    def _run_five_pt_algorithm(self, tracks, prev_R, prev_T):
        five_pt_config = self.config.five_point_algorithm
        recover_pose_config = self.config.recover_pose_algorithm

        if len(tracks[0]) < five_pt_config.min_number_of_points:
            return None, None, None, None

        # We have no 3D point info so we calculate based on the two cameras
        E, five_pt_mask = cv2.findEssentialMat(
            points1=tracks[0],
            points2=tracks[1],
            cameraMatrix=self.config.camera_matrix,
            method=cv2.RANSAC,
            prob=five_pt_config.probability,
            threshold=five_pt_config.threshold,
            # mask=track_mask,
        )

        # print(
        #     f"E: {sum(five_pt_mask.squeeze()):3}/{five_pt_mask.shape[0]:3}",
        #     end="\t",
        # )

        _, R, T, pose_mask, points_4d = cv2.recoverPose(
            E=E,
            points1=tracks[0],
            points2=tracks[1],
            cameraMatrix=self.config.camera_matrix,
            distanceThresh=recover_pose_config.distance_threshold,
            mask=five_pt_mask.copy(),
        )

        # print(f"P: {sum(pose_mask.squeeze()):3}/{pose_mask.shape[0]:3}")

        # filter out 3d_points and point_indexes according to mask
        final_mask = pose_mask.squeeze().astype(np.bool)

        points_3d = cv2.convertPointsFromHomogeneous(
            points_4d.transpose()
        ).squeeze()[final_mask]

        # Convert it back to first camera base system
        R, T = utils.invert_reference_frame(R, T)

        # Then convert it all to camera 0's reference system
        points_3d = utils.translate_points_to_base_frame(
            prev_R, prev_T, points_3d
        )
        R, T = utils.compose_rts(R, T, prev_R, prev_T)

        return R, T, points_3d, final_mask

    def _reproject_tracks_to_3d(self, R_1, T_1, R_2, T_2, tracks):
        if any([R_1 is None, T_1 is None, R_2 is None, T_2 is None,]):
            return None

        assert (
            tracks.shape[0] == 2
        ), "Can't do reprojection with {} cameras, 2 are needed".format(
            tracks.shape[0]
        )

        P1 = np.matmul(self.config.camera_matrix, np.hstack((R_1, T_1)))
        P2 = np.matmul(self.config.camera_matrix, np.hstack((R_2, T_2)))

        points_4d = cv2.triangulatePoints(
            projMatr1=P1,
            projMatr2=P2,
            projPoints1=tracks[0].transpose(),
            projPoints2=tracks[1].transpose(),
        )
        points_3d = cv2.convertPointsFromHomogeneous(
            points_4d.transpose()
        ).squeeze()

        return points_3d

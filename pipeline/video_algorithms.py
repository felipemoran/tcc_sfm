import cv2
import numpy as np
from os import path
import os

from pipeline import utils

debug_colors = np.random.randint(0, 255, (200, 3))


def klt_generator(config, file):
    reset_features = True

    prev_frame = None
    prev_features = np.empty((0, 2), dtype=np.float_)
    features = np.empty((0, 2), dtype=np.float_)
    indexes = np.empty((0,))
    start_index = 100

    for counter, (color_frame, bw_frame) in enumerate(
        frame_skipper(file, config.frames_to_skip)
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

            features, indexes = match_features(
                features, indexes, new_features, start_index
            )
            start_index = max(indexes) + 1

        yield features, indexes
        prev_frame, prev_features = bw_frame, features

        if config.display_klt_debug_frames:
            display_klt_debug_frame(
                color_frame, features, prev_features, indexes,
            )


def match_features(old_features, old_indexes, new_features, index_start):
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


def frame_skipper(file, frames_to_skip):
    while True:
        for _ in range(frames_to_skip + 1):
            ret, color_frame = file.read()
            if not ret:
                return
        bw_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)
        yield color_frame, bw_frame


def get_video(file_path):
    print("Looking for files in {}".format(dir))

    assert path.isfile(file_path), "Invalid file location"

    file = cv2.VideoCapture(file_path)
    filename = os.path.basename(file_path)

    return file, filename


def display_klt_debug_frame(
    color_frame, features, prev_features, indexes,
):
    mask = np.zeros_like(color_frame)

    for feature, prev_feature, index in zip(features, prev_features, indexes):
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
            color_frame, tuple(feature), 5, debug_colors[index].tolist(), -1,
        )

    img = cv2.add(color_frame, mask)

    cv2.imshow("frame", img)
    cv2.waitKey(1)

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
        frame_getter(file, config.frames_to_skip)
    ):
        if prev_frame is not None:
            features, indexes = track_features(
                bw_frame, config, indexes, prev_features, prev_frame
            )

        if reset_features or counter % config.reset_period == 0:
            reset_features = False

            features, indexes = get_new_features(
                bw_frame, config, features, indexes, start_index
            )
            start_index = max(indexes) + 1

        yield features, indexes
        prev_frame, prev_features = bw_frame, features

        if config.display_klt_debug_frames:
            display_klt_debug_frame(
                color_frame,
                features,
                prev_features,
                indexes,
                config.klt_debug_frames_delay,
            )


def get_new_features(bw_frame, config, features, indexes, start_index):
    new_features = cv2.goodFeaturesToTrack(
        image=bw_frame,
        maxCorners=config.max_features,
        qualityLevel=config.corner_selection.quality_level,
        minDistance=config.corner_selection.min_distance,
        mask=None,
        blockSize=config.corner_selection.block_size,
    ).squeeze()
    features, indexes = match_features(
        features,
        indexes,
        new_features,
        start_index,
        config.closeness_threshold,
        config.max_features,
    )
    return features, indexes


def track_features(bw_frame, config, indexes, prev_features, prev_frame):
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
    return features, indexes


def match_features(
    old_features,
    old_indexes,
    new_features,
    index_start,
    threshold,
    max_features,
):
    if len(old_features) == 0:
        return new_features, np.arange(len(new_features)) + index_start

    closeness_table = get_close_points_table(
        new_features, old_features, threshold
    )

    new_points_mask = closeness_table.sum(axis=0) == 0

    new_features = new_features[new_points_mask]
    new_indexes = np.arange(len(new_features)) + index_start

    # TODO: limit number of returned features
    points_to_keep = min(max_features - len(new_features), len(old_features))
    old_features_mask = choose_old_features(closeness_table, points_to_keep)

    old_features = old_features[old_features_mask]
    old_indexes = old_indexes[old_features_mask]

    features = np.vstack((old_features, new_features))
    indexes = np.concatenate((old_indexes, new_indexes))

    assert len(features) == len(indexes)

    return features, indexes


def choose_old_features(closeness_table, points_to_keep):
    mask = np.full(closeness_table.shape[0], False)
    indexes = np.empty([0], dtype=int)
    table_sum = closeness_table.sum(axis=1)

    base_indexes = np.arange(len(mask))

    # TODO: continue debugging here

    for sum_threshold in range(max(table_sum + 1)):
        points_to_go = points_to_keep - len(indexes)
        threshold_mask = table_sum == sum_threshold

        if sum(threshold_mask) <= points_to_go:
            indexes = np.hstack((indexes, base_indexes[threshold_mask]))
        else:
            indexes = np.hstack(
                (
                    indexes,
                    np.random.choice(
                        base_indexes[threshold_mask],
                        points_to_go,
                        replace=False,
                    ),
                )
            )

        assert len(indexes) <= points_to_keep

        if len(indexes) == points_to_keep:
            mask[indexes] = True
            break

    return mask


def get_close_points_table(new_features, old_features, threshold):
    old_repeated = np.repeat(
        old_features, new_features.shape[0], axis=0
    ).reshape((old_features.shape[0], new_features.shape[0], 2))
    new_repeated = (
        np.repeat(new_features, old_features.shape[0], axis=0)
        .reshape((new_features.shape[0], old_features.shape[0], 2))
        .transpose((1, 0, 2))
    )
    distances = np.linalg.norm(old_repeated - new_repeated, axis=2)
    close_points = distances < threshold
    return close_points


def frame_getter(file, frames_to_skip):
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
    color_frame, features, prev_features, indexes, delay
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
            debug_colors[index % 200].tolist(),
            2,
        )
        color_frame = cv2.circle(
            color_frame,
            tuple(feature),
            5,
            debug_colors[index % 200].tolist(),
            -1,
        )

    img = cv2.add(color_frame, mask)

    cv2.imshow("frame", img)
    cv2.waitKey(delay)

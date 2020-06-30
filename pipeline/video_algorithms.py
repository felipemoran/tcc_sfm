import cv2
import numpy as np
from os import path
import os

from pipeline import utils

debug_colors = np.random.randint(0, 255, (200, 3))


def klt_generator(config, file):
    """
    Generates and tracks KLT features from a video file.

    This function generates new features according to the parameters set in config and tracks
    old features as the video progresses. At each frame it tracks previous features
    and according to the number of frames to skip it yields a vector of features and their
    indexes.

    Feature indexes are unique and never repeating therefore can be used for indexing throughout
    the pipeline without collision.

    :param config: config object. See config.py for more information
    :param file: video file object
    :return: generator of tracks. Each item is of the form
        (
            frame_number: number of frame, monotonically increasing
            features: list of tuples with each feature location
            indexes: list of indexes for each returned features. These indexes
            are a global non repeating number corresponding to each feature
            and can be used to uniquely reference a reconstructed point
        )
    """

    reset_features = True

    prev_frame = None
    prev_features = np.empty((0, 2), dtype=np.float_)
    features = np.empty((0, 2), dtype=np.float_)
    indexes = np.empty((0,))
    start_index = 100

    if config.calculate_every_frame:
        reset_period = config.reset_period * (config.frames_to_skip + 1)
        skip_at_getter = 0
        yield_period = config.frames_to_skip + 1
    else:
        reset_period = config.reset_period
        skip_at_getter = config.frames_to_skip
        yield_period = 1

    for frame_number, (color_frame, bw_frame) in enumerate(
        frame_getter(file, skip_at_getter)
    ):
        if prev_frame is not None:
            features, indexes = track_features(
                bw_frame, config, indexes, prev_features, prev_frame
            )

        if reset_features or frame_number % reset_period == 0:
            reset_features = False

            features, indexes = get_new_features(
                bw_frame, config, features, indexes, start_index
            )
            start_index = max(indexes) + 1

        if frame_number % yield_period == 0:
            yield frame_number, features, indexes

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
    """
    Calculates new features to track and mergest it with previous ones

    :param bw_frame: frame in black and white
    :param config: config object. See config.py for more information
    :param features: previous features
    :param indexes: previous features indexes
    :param start_index: next index to be used on new features
    :return: new set of features to be tracked and corresponding indexes
    """

    new_features = cv2.goodFeaturesToTrack(
        image=bw_frame,
        maxCorners=config.max_features,
        qualityLevel=config.corner_selection.quality_level,
        minDistance=config.corner_selection.min_distance,
        mask=None,
        blockSize=config.corner_selection.block_size,
    ).reshape((-1, 2))

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
    """
    Tracks a given set of indexes returning their new positions and dropping features
    that are not found

    :param bw_frame: frame in black and white
    :param config: config object. See config.py for more information
    :param indexes: previous features indexes
    :param prev_features: features from previous frames
    :param prev_frame: previous frame
    :return: new location of features and indexes
    """

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
    indexes = indexes[status].reshape((-1,))
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
    """
    Given a set of old features and new proposed features, it selects which features to use.

    Selection is done by calculating the euclidean distance between all features from
    old_features vector 

    :param old_features:
    :param old_indexes:
    :param new_features:
    :param index_start:
    :param threshold:
    :param max_features:
    :return:
    """
    if len(old_features) == 0:
        return new_features, np.arange(len(new_features)) + index_start

    closeness_table = get_close_points_table(
        new_features, old_features, threshold
    )

    new_points_mask = closeness_table.sum(axis=0) == 0

    new_features = new_features[new_points_mask]
    new_indexes = np.arange(len(new_features)) + index_start

    # limit number of returned features
    points_to_keep = min(max_features - len(new_features), len(old_features))
    old_features_mask = choose_old_features(closeness_table, points_to_keep)

    old_features = old_features[old_features_mask]
    old_indexes = old_indexes[old_features_mask]

    features = np.vstack((old_features, new_features))
    indexes = np.concatenate((old_indexes, new_indexes))

    assert len(features) == len(indexes)

    return features, indexes


def choose_old_features(closeness_table, points_to_keep):
    """
    Based on the number of close features (from closeness_table) it chooses which features
    to keep and which to discard

    :param closeness_table: table of 0s and 1s where a 1 indicates that the ith point
    from the old feature set is close enough to the jth point from the new feature set
    :param points_to_keep: number of points from old feature set to keep
    :return: mask of bools indicating which features to keep
    """

    mask = np.full(closeness_table.shape[0], False)
    indexes = np.empty([0], dtype=int)
    table_sum = closeness_table.sum(axis=1)

    base_indexes = np.arange(len(mask))

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
    """
    Generated the closeness table of 0s and 1s where a 1 indicates that the ith point
    from the old feature set is close enough to the jth point from the new feature set.

    The size of the table is len(old_features) lines and len(new_features) columns.

    :param new_features: list of new features to track
    :param old_features: list of old features to track
    :param threshold: distance (in pixels) below which two points are considered close
    :return:
    """
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
    """
    Gets a frame from file, skips frames_to_skip times and returns the color and
    grayscale version of next frame

    :param file: video file
    :param frames_to_skip: number of frames to skip between returned frames
    :return: original video frame and grayscale version
    """
    while True:
        for _ in range(frames_to_skip + 1):
            ret, color_frame = file.read()
            if not ret:
                return
        bw_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)
        yield color_frame, bw_frame


def get_video(file_path):
    """
    Loads file from file_path

    :param file_path:
    :return:
    """
    # print("Looking for files in {}".format(dir))

    assert path.isfile(file_path), "Invalid file location"

    file = cv2.VideoCapture(file_path)
    filename = os.path.basename(file_path)

    return file, filename


def display_klt_debug_frame(
    color_frame, features, prev_features, indexes, delay
):
    """
    Display a frame and colored dots indicating which features are being tracked

    :param color_frame: original frame
    :param features: list of features being tracked
    :param prev_features: list of previous feature positions
    :param indexes: list of featue indexes
    :param delay: wait time before exiting visualization
    :return: None
    """
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

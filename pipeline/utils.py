import os
from itertools import count, chain
from dataclasses import dataclass

import numpy as np


TYPE_CALIBRATION_MATRIX = 0
TYPE_CAMERA = 1
TYPE_POINT = 2


@dataclass
class ErrorMetric:
    frame_number: int
    projection: float
    cam_orientation: float
    cam_position: float
    point_position: float


def write_to_viz_file(camera_matrix, Rs, Ts, points):
    """
    Writes input information as a properly formatted csv file for visualization

    :param camera_matrix: calibration matrix
    :param Rs: list of R matrices
    :param Ts: list of T vectors
    :param points: point cloud with N points as a ndarray with shape Nx3
    """
    # IMPORTANT: Rs and ts must be in the global coordinate system

    with open("out/viz_data.csv", "w") as out_file:
        out_file.write("{}\n")

        def convert_and_save_line(line):
            line = [str(item) for item in line]
            out_file.write(",".join(line))
            out_file.write("\n")

        line_elements = [TYPE_CALIBRATION_MATRIX, 0] + list(
            camera_matrix.flatten()
        )
        convert_and_save_line(line_elements)

        for index, (R, t) in enumerate(zip(Rs, Ts)):
            line_elements = (
                [TYPE_CAMERA, index] + list(R.flatten()) + list(t.flatten())
            )
            convert_and_save_line(line_elements)

        for point_id, point in enumerate(points):
            if np.isnan(point).any():
                continue

            line_elements = [TYPE_POINT, point_id] + list(point.flatten())
            # if 'color' in point:
            #     line_elements += list(point['point_color'].flatten())

            convert_and_save_line(line_elements)


def call_viz():
    """
    Call visualizes with default csv file location
    """
    os.system(
        os.path.join(
            os.getcwd(), "visualizer", "cmake-build-debug", "visualizer"
        )
        + " "
        + os.path.join(os.getcwd(), "out", "viz_data.csv")
    )


def visualize(camera_matrix, Rs, Ts, points):
    """
    Bundles together visualization file creation and visualizer call

    :param camera_matrix: calibration matrix
    :param Rs: list of R matrices
    :param Ts: list of T vectors
    :param points: point cloud with N points as a ndarray with shape Nx3
    """
    write_to_viz_file(camera_matrix, Rs, Ts, points)
    call_viz()


def compose_rts(rel_R, rel_T, comp_R, comp_T):
    """
    Porperly composes two sets of rotation matrices and translation vectors

    :param rel_R: rotation matrix of new camera
    :param rel_T: translation vector of new camera
    :param comp_R: rotation matrix of previous camera
    :param comp_T: translation vector of previous camera
    :return: composed rotation matrix and translation vector
    """
    res_T = comp_T + np.matmul(comp_R, rel_T)
    res_R = np.matmul(comp_R, rel_R)
    return res_R, res_T


def translate_points_to_base_frame(comp_R, comp_T, points):
    """
    Convert point from local reference frame to global's
    :param comp_R: camera's rotation matrix in the global reference frame
    :param comp_T: camera's translation vector in the global reference frame
    :param points: points to be converted
    :return: point coordinates in the global reference frame
    """
    return (comp_T + np.matmul(comp_R, points.transpose())).transpose()


def get_nan_bool_mask(input_array):
    """
    Creates mask of booleans indicating which values are Nan/None

    :param input_array: feature track or point cloud
    :return: bool mask
    """
    return (np.isnan(input_array)).any(axis=1)


def get_nan_index_mask(input_array):
    """
    Creates mask with the indexes of elements that are Nan/None.

    :param input_array: feature track or point cloud
    :return: index mask
    """
    nan_bool_mask = get_nan_bool_mask(input_array)
    return np.arange(len(input_array))[nan_bool_mask]


def get_not_nan_index_mask(input_array):
    """
    Creates mask with the indexes of elements that are not Nan/None.

    :param input_array: feature track or point cloud
    :return: index mask
    """

    not_nan_bool_mask = ~get_nan_bool_mask(input_array)
    return np.arange(len(input_array))[not_nan_bool_mask]


def invert_reference_frame(R, T):
    """
    Inverts rotation matrix and translation vector

    :param R: rotation matrix
    :param T: translation vector
    :return: inverted R and T
    """
    if R is None:
        return T, R
    return R.transpose(), np.matmul(R.transpose(), -T)


def init_rt():
    """
    Creates initial rotation matrix and translation vector

    :return: 3x3 identity matix and vector of zeros of shape 3x1
    """
    return np.eye(3), np.zeros((3, 1), dtype=np.float_)


def get_intersection_mask(a, b):
    """
    Calculates the intersection of two index returning only those that are present in both
    vectors a nd b

    :param a: index vector a
    :param b: index vector b
    :return: intesection between a and b
    """
    return np.intersect1d(a, b)


def get_last_track_pair(tracks, masks):
    """
    Forms a track pair with the last two track slices.

    This formed pair consists of all the features that are present in both traacks.

    :param tracks: list of 2D feature vectors. Each vector has the shape Dx2
    :param masks: list of index masks for each feature vector. Indexes refer to the position of
    the item in the cloud
    :return: track pair and pair index mask in the same structure as tracks and masks
    """
    pair_mask = get_intersection_mask(masks[-2], masks[-1])
    track_pair = [
        # track[[item in pair_mask for item in mask]]
        track[np.isin(mask, pair_mask)]
        for (track, mask) in zip(tracks[-2:], masks[-2:])
    ]
    return track_pair, pair_mask


def points_to_cloud(points, indexes):
    """
    Creates a cloud of points from a set of sparse 3D points and indexes

    :param points: point cloud with N points as a ndarray with shape Nx3
    :param indexes: list of index masks for each feature vector. Indexes refer to the position
    of the item in the cloud
    :return: point cloud
    """
    cloud = np.full((max(indexes) + 1, 3), None, dtype=np.float_)
    cloud[indexes] = points
    return cloud


def add_points_to_cloud(cloud, points, index_mask):
    """
    Adds new points from 'points' to the cloud

    :param cloud: point cloud with N points as a ndarray with shape Nx3
    :param points: calculated 3D points from a given set of frames
    :param index_mask: vector with corresponding indexes for each point in points
    :return:
    """

    assert cloud is not None

    cloud_mask = get_not_nan_index_mask(cloud)
    new_points_mask = np.setdiff1d(index_mask, cloud_mask)

    if max(index_mask) >= cloud.shape[0]:
        new_cloud = np.full((max(index_mask) * 2, 3), None, dtype=np.float_)
        new_cloud[cloud_mask] = cloud[cloud_mask]
        cloud = new_cloud

    cloud[new_points_mask] = points[np.isin(index_mask, new_points_mask)]

    return cloud


def generator_copier(generator, num_copies, num_elements=None):
    """
    Copies a generator multiple times up to a certain point or until it's end,
    whichever comes first. The generator evaluation is eager, meaning that if an
    infinite generator is passed without a number of elements to be copied the
    code will hang.

    :param generator: generator to be copied
    :param num_copies: number of copies to be generated
    :param num_elements: number of elements to be copied
    :return: list with original generator and copies
    """
    return_list = []

    index_generator = (
        range(num_elements) if num_elements is not None else count(0, 1)
    )

    for _ in index_generator:
        try:
            return_list += [next(generator)]
        except StopIteration:
            break

    generators = [chain(return_list, generator)] + [
        (item for item in return_list) for _ in range(num_copies)
    ]

    generators[0] = chain(generators[0], generator)

    return generators

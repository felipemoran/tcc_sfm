import os
import numpy as np

TYPE_CALIBRATION_MATRIX = 0
TYPE_CAMERA = 1
TYPE_POINT = 2


def write_to_viz_file(camera_matrix, Rs, Ts, points):
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
    os.system(
        os.path.join(
            os.getcwd(), "visualizer", "cmake-build-debug", "visualizer"
        )
        + " "
        + os.path.join(os.getcwd(), "out", "viz_data.csv")
    )


def progress_bar(realized, total, length=20):
    assert 0 <= realized <= total

    rep = int(round(length * realized / total))
    return "{:2} / {:2}  {:2}".format(
        realized, total, "#" * rep + "_" * (length - rep)
    )


def compose_RTs(rel_R, rel_T, comp_R, comp_T):
    res_T = comp_T + np.matmul(comp_R, rel_T)
    res_R = np.matmul(comp_R, rel_R)
    return res_R, res_T


def translate_points_to_base_frame(comp_R, comp_T, points):
    return (comp_T + np.matmul(comp_R, points.transpose())).transpose()


def get_nan_mask(input_array):
    return (np.isnan(input_array)).any(axis=1)


def invert_reference_frame(R, T):
    if R is None:
        return T, R
    return R.transpose(), np.matmul(R.transpose(), -T)

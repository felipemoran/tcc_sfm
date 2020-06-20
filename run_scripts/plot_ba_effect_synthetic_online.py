import time
from functools import reduce

import dacite
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pipeline import utils
from pipeline.config import VideoPipelineConfig
from pipeline.synthetic_pipeline import SyntheticPipeline
from pipeline.video_pipeline import VideoPipeline

from ruamel.yaml import YAML

config_string = """
pipeline_type: "synthetic"
file_path: "3"

synthetic_config:
    noise_covariance: 5
    number_of_cameras: 500

    case_3:
        radius: 5
        number_of_cameras: 25
        step_size: 0.5
        x_points: 5
        y_points: 5
        z_points: 4

camera_matrix: &camera_matrix [
    [765.16859169, 0.0, 379.11876567],
    [0.0, 762.38664643, 497.22086655],
    [0.0, 0.0, 1.0],
]

use_five_pt_algorithm: true
use_solve_pnp: true
use_reconstruct_tracks: true

klt:
    calculate_every_frame: true
    display_klt_debug_frames: true
    klt_debug_frames_delay: 1
    frames_to_skip: 0
    reset_period: 1
    closeness_threshold: 15
    max_features: 100

    corner_selection:
        quality_level: 0.5
        min_distance: 15
        block_size: 10

    optical_flow:
        window_size:
            width: 15
            height: 15
        max_level: 3
        criteria:
            # original: (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
            # where cv2.TERM_CRITERIA_EPS = 2 and cv2.TERM_CRITERIA_COUNT = 1
            criteria_sum: 3
            max_iter: 30
            eps: 0.003

error_calculation:
    period: 1
    window_length: 1
    online_calculation: True
    post_calculation: True

bundle_adjustment:
    tol: 1e-2
    method: "trf"
    verbose: 0
    camera_matrix: *camera_matrix

    use_with_rolling_window: false
    rolling_window:
        method: growing_step
        period: 1 # in number of processed frame
        length: 6 # number of frames in the window taking step into account
        step: 1 # step=N means taking 1 every N frames
#    rolling_window:
#        method: constant_step
#        period: 5 # in number of processed frame
#        length: 10 # number of frames in the window taking step into account
#        step: 1 # step=N means taking 1 every N frames
    use_at_end: false

five_pt_algorithm:
    min_number_of_points: 6
    essential_mat_threshold: 5
    ransac_probability: 0.999999
    refine_matches_repetitions: 1
    save_optimized_projections: false
    camera_matrix: *camera_matrix
    distance_threshold: 500

solve_pnp:
    min_number_of_points: 5
    camera_matrix: *camera_matrix
    use_epnp: true
    use_iterative_pnp: true

init:
#    method: five_pt_algorithm
    error_threshold: 25
    num_reconstruction_frames: 5
    num_error_calculation_frames: 5

"""


def generate_data():
    num_runs = 50

    yaml = YAML()

    df_init_errors = None
    df_online_errors = None
    df_post_errors = None

    case_labels = [
        "Rolling window: OFF",
        "Roliing window: ON",
    ]

    for case in range(len(case_labels)):
        config_raw = yaml.load(config_string)
        config = dacite.from_dict(
            data=config_raw, data_class=VideoPipelineConfig
        )

        start = time.time()

        pipeline = SyntheticPipeline(config=config)

        if case == 0:
            config.bundle_adjustment.use_with_rolling_window = False
            config.bundle_adjustment.use_at_end = True
        elif case == 1:
            config.bundle_adjustment.use_with_rolling_window = True
            config.bundle_adjustment.use_at_end = True
        else:
            raise ValueError()

        total_errors = []

        for run in range(num_runs):
            print(
                f"--- Case {case+1}/{len(case_labels)} - Run {run + 1}/{num_runs}"
            )
            Rs, Ts, cloud, init_error, online_error, post_error = pipeline.run()

            df_init_error = pd.DataFrame([x.__dict__ for x in init_error])
            df_init_error["case"] = f"{case_labels[case]}"
            df_init_error["run"] = run

            df_online_error = pd.DataFrame([x.__dict__ for x in online_error])
            df_online_error["case"] = f"{case_labels[case]}, BA at end: OFF"
            df_online_error["run"] = run

            df_post_error = pd.DataFrame([x.__dict__ for x in post_error])
            df_post_error["case"] = f"{case_labels[case]}, BA at end: ON"
            df_post_error["run"] = run

            if df_init_error is None:
                df_init_errors = df_init_error
                df_online_errors = df_online_error
                df_post_errors = df_post_error
            else:
                df_init_errors = pd.concat([df_init_errors, df_init_error])
                df_online_errors = pd.concat(
                    [df_online_errors, df_online_error]
                )
                df_post_errors = pd.concat([df_post_errors, df_post_error])

        elapsed = time.time() - start
        print("Elapsed {}".format(elapsed))

        # utils.visualize(config.camera_matrix, Rs, Ts, cloud)

        #     POST PROCESSING

        df_reconstruction = pd.concat([df_online_errors, df_post_errors])

    return [df_init_errors, df_reconstruction]


def plot_data(dfs):
    plt.figure()
    sns.barplot(x="case", y="dropped_frames", data=df_drop)

    for df, title in zip(dfs, ["init", "reconstruction"],):
        fig, axes = plt.subplots(2, 2)
        fig.suptitle(title)

        for column, p in zip(
            ["cam_orientation", "cam_position", "point_position", "projection"],
            [(0, 0), (0, 1), (1, 0), (1, 1)],
        ):
            sns.lineplot(
                x="frame_number",
                y=column,
                data=df,
                hue="case",
                ax=axes[p[0], p[1]],
                # ci=None,
            )
    plt.show()


def save_data(data):
    with open("out/ba_synthetic_data.pkl", "wb") as f:
        pickle.dump(data, f)


def load_data():
    with open("out/ba_synthetic_data.pkl", "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    # data_1 = generate_data()
    #
    # save_data(data_1)

    data_2 = load_data()

    plot_data(data_2)

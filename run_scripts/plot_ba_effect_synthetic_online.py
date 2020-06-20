import time
from functools import reduce

import dacite
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pipeline import utils
from pipeline.config import VideoPipelineConfig
from pipeline.synthetic_pipeline import SyntheticPipeline
from pipeline.video_pipeline import VideoPipeline

from ruamel.yaml import YAML

config_string = """
pipeline_type: "synthetic"
file_path: "3"

synthetic_config:
    noise_covariance: 10
    number_of_cameras: 500

    case_3:
        radius: 8
        number_of_cameras: 50
        size: 4

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
    error_threshold: 85
    num_reconstruction_frames: 10
    num_error_calculation_frames: 5

"""


if __name__ == "__main__":
    yaml = YAML()

    case_errors = []

    labels = [
        "1: no BA",
        "3: rolling window BA",
    ]

    for case in range(len(labels)):
        config_raw = yaml.load(config_string)
        config = dacite.from_dict(
            data=config_raw, data_class=VideoPipelineConfig
        )

        start = time.time()

        pipeline = SyntheticPipeline(config=config)

        if case == 0:
            config.bundle_adjustment.use_with_rolling_window = False
        elif case == 1:
            config.bundle_adjustment.use_with_rolling_window = True
        else:
            raise ValueError()

        total_errors = []

        for i in range(250):
            print(f"--- Run {i}")
            Rs, Ts, cloud, errors = pipeline.run()
            total_errors += [errors]

        elapsed = time.time() - start
        print("Elapsed {}".format(elapsed))

        print(f"Errors: {errors}")

        # utils.visualize(config.camera_matrix, Rs, Ts, cloud)

        #     POST PROCESSING

        case_errors += reduce(
            lambda a, b: a + b,
            [
                [
                    [case, run_number, counter] + item
                    for counter, item in enumerate(run_errors)
                ]
                for run_number, run_errors in enumerate(total_errors)
            ],
        )

    df = pd.DataFrame(
        np.array(case_errors), columns=["case", "run", "counter", "r", "t", "p"]
    )

    fig, axis = plt.subplots(2, 2)

    labels = [
        "1: no BA",
        "3: rolling window BA",
    ]

    for index, case in df.groupby("case"):
        axis[0, 0].plot(
            case[case.run == 0].counter,
            case.groupby("counter").r.mean(),
            label=labels[int(index)],
        )
        axis[0, 0].fill_between(
            case[case.run == 0].counter,
            case.groupby("counter").r.mean() - case.groupby("counter").r.std(),
            case.groupby("counter").r.mean() + case.groupby("counter").r.std(),
            alpha=0.25,
        )

        axis[0, 1].plot(
            case[case.run == 0].counter, case.groupby("counter").t.mean(),
        )
        axis[0, 1].fill_between(
            case[case.run == 0].counter,
            case.groupby("counter").t.mean() - case.groupby("counter").t.std(),
            case.groupby("counter").t.mean() + case.groupby("counter").t.std(),
            alpha=0.25,
        )

        axis[1, 0].plot(
            case[case.run == 0].counter, case.groupby("counter").p.mean(),
        )
        axis[1, 0].fill_between(
            case[case.run == 0].counter,
            case.groupby("counter").p.mean() - case.groupby("counter").p.std(),
            case.groupby("counter").p.mean() + case.groupby("counter").p.std(),
            alpha=0.25,
        )

    axis[0, 0].title.set_text("camera angle error (degrees)")
    axis[0, 1].title.set_text("camera position error")
    axis[1, 0].title.set_text("point position error")

    fig.legend()

    plt.show()

    df.to_pickle("out/result_df.pkl")

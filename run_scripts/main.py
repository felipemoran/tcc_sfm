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

if __name__ == "__main__":
    yaml = YAML()

    with open("config.yaml", "r") as f:
        config_raw = yaml.load(f)
    config = dacite.from_dict(data=config_raw, data_class=VideoPipelineConfig)

    start = time.time()

    if config.pipeline_type == "synthetic":
        pipeline = SyntheticPipeline(config=config)
    elif config.pipeline_type == "video":
        pipeline = VideoPipeline(config=config)
    else:
        raise Exception("Invalid pipeline type")

    Rs, Ts, cloud, online_errors, post_errors = pipeline.run()

    elapsed = time.time() - start
    print("Elapsed {}".format(elapsed))

    print(f"Errors: {online_errors}")
    print(f"Errors: {post_errors}")

    df = pd.DataFrame([x.__dict__ for x in post_errors])
    print(df)
    utils.visualize(config.camera_matrix, Rs, Ts, cloud)
    a = 1

import time
from functools import reduce

import dacite
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

    (
        Rs,
        Ts,
        cloud,
        init_errors,
        online_errors,
        post_errors,
        execution_time,
    ) = pipeline.run()

    elapsed = time.time() - start
    print("Elapsed {}".format(elapsed))

    print(f"Errors: {online_errors}")
    print(f"Errors: {post_errors}")

    dfs = [
        pd.DataFrame([x.__dict__ for x in errors])
        for errors in [init_errors, online_errors, post_errors]
    ]

    df_combined = pd.DataFrame()

    for df, label in zip(dfs, ["init", "online", "post"]):
        df["type"] = label
        df_combined = pd.concat([df_combined, df])

    print(df)
    print(f"Execution time: {execution_time}")
    plt.figure()
    sns.lineplot(x="frame_number", y="projection", hue="type", data=df_combined)
    plt.figure()
    sns.lineplot(
        x="frame_number",
        y="projection",
        hue="type",
        data=df_combined[df_combined.type == "online"],
    )
    plt.figure()
    sns.lineplot(
        x="frame_number",
        y="projection",
        hue="type",
        data=df_combined[df_combined.type == "post"],
    )

    plt.show()
    utils.visualize(config.camera_matrix, Rs, Ts, cloud)

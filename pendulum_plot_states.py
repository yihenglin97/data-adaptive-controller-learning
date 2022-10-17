import itertools as it
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def main():
    systems = ["linear", "nonlinear"]
    noises = ["gaussian", "walk"]
    #datas = [np.load(f"pendulum_{n}.npz") for n in names]
    ANGLE = "angle (degrees)"

    dfs = []
    for system, noise in it.product(systems, noises):
        data = np.load(f"pendulum_{system}_{noise}.npz")
        dt = data["dt"]
        x_log = data["x_log"]
        time = dt * np.arange(len(x_log))
        for i, controller in enumerate(["ours", "LQ"]):
            dfs.append(pd.DataFrame({
                "time": time,
                ANGLE: np.degrees(x_log[:, i, 0]),
                "controller": controller,
                "disturbance": noise,
                "system": system,
            }))
    df = pd.concat(dfs, ignore_index=True)

    grid = sns.relplot(
        data=df,
        kind="line",
        row="system",
        col="disturbance",
        x="time",
        y=ANGLE,
        hue="controller",
        height=2.5,
        aspect=2.0,
        facet_kws=dict(
            gridspec_kws=dict(
                hspace=0.05,
            )
        )
    )
    # Python's sloppy scoping - `time` was defined in loading loop.
    grid.set(xlim=[time[0], time[-1] + dt])
    grid.savefig("pendulum_states.pdf")


if __name__ == "__main__":
    main()

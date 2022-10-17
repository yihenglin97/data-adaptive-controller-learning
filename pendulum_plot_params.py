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

    dfs = []
    for system, noise in it.product(systems, noises):
        data = np.load(f"pendulum_{system}_{noise}.npz")
        dt = data["dt"]
        for controller in ["ours", "LQ"]:
            theta_log = data["theta_log_" + controller]
            time = dt * np.arange(len(theta_log))
            for value, paramname in zip(theta_log.T, ["kp", "kd"]):
                dfs.append(pd.DataFrame(dict(
                    time=time,
                    gain=value,
                    param=paramname,
                    controller=controller,
                    disturbance=noise,
                    system=system,
                )))
    df = pd.concat(dfs, ignore_index=True)

    sns.set_style("ticks", {"axes.grid" : True})
    param_grid = sns.relplot(
        data=df,
        kind="line",
        col="disturbance",
        row="system",
        x="time",
        y="gain",
        style="controller",
        hue="param",
        height=2.5,
        aspect=2.0,
        facet_kws=dict(
            gridspec_kws=dict(
                hspace=0.05,
            )
        )
    )
    # Python's sloppy scoping - `time` was defined in loading loop.
    param_grid.set(xlim=[time[0], time[-1] + dt])
    param_grid.savefig("pendulum_params.pdf")


if __name__ == "__main__":
    main()

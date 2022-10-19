import itertools as it
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


REGRET = "cost difference, ours - LQR"


def main():

    plt.rc("text", usetex=True)
    plt.rc("font", size=12)

    noises = ["gaussian", "walk"]

    dfs = []
    for noise in noises:
        data = np.load(f"pendulum_nonlinear_{noise}.npz")
        dt = data["dt"]
        cost_log = data["cost_log"]
        time = dt * np.arange(len(cost_log))
        cost_cumulative = np.cumsum(cost_log, axis=0).T
        regret = (cost_cumulative[0] - cost_cumulative[1]).squeeze()
        dfs.append(pd.DataFrame({
            "time": time,
            REGRET: regret,
            "time": time,
            "disturbance": noise,
        }))
    df = pd.concat(dfs, ignore_index=True)

    sns.set_style("ticks", {"axes.grid" : True})
    grid = sns.relplot(
        data=df,
        kind="line",
        x="time",
        y=REGRET,
        col="disturbance",
        color="black",
        height=2.7,
        aspect=2.2,
        facet_kws=dict(
            gridspec_kws=dict(
                hspace=0.05,
            )
        )
    )
    # Special case for main result in paper.
    ylim = list(grid.axes.flat[0].get_ylim())
    if -8200 < ylim[0] < -8000:
        ylim[0] = -8000
    # Python's sloppy scoping - `time` was defined in loading loop.
    grid.set(xlim=[time[0], time[-1] + dt], ylim=ylim)
    grid.savefig("pendulum_costs.pdf")


if __name__ == "__main__":
    main()

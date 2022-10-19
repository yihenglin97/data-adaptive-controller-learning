import itertools as it
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def main():
    noises = ["gaussian", "walk"]
    ANGLE = "angle (degrees)"

    dfs = []
    for noise in noises:
        data = np.load(f"pendulum_nonlinear_{noise}.npz")
        dt = data["dt"]
        x_log = data["x_log"]
        time = dt * np.arange(len(x_log))
        for i, controller in enumerate(["ours", "LQ"]):
            dfs.append(pd.DataFrame({
                "time": time,
                ANGLE: np.degrees(x_log[:, i, 0]),
                "controller": controller,
                "disturbance": noise,
            }))
    df = pd.concat(dfs, ignore_index=True)

    grid = sns.relplot(
        data=df,
        kind="line",
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
    for ax in grid.axes.flat:
        ax.grid(True)
    # Zoom in to the outermost ticks - it's ok if we clip a few stray peaks.
    ax0 = grid.axes.flat[0]
    ylim = np.amax(np.abs(ax0.get_ylim()))
    tickabs = np.abs(ax0.get_yticks())
    ymax = np.amax(tickabs[tickabs < ylim])
    # Python's sloppy scoping - `time` was defined in loading loop.
    grid.set(xlim=[time[0], time[-1] + dt], ylim=[-ymax, ymax])
    grid.savefig("pendulum_states.pdf")


if __name__ == "__main__":
    main()

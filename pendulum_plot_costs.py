import itertools as it
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


REGRET = "cost(ours) - cost(LQR), cumulative"


def main():
    names = ["gaussian", "walk"]
    datas = [np.load(f"pendulum_{n}.npz") for n in names]
    colors = ["red", "black"]

    fig, ax = plt.subplots(1, 1, figsize=(5, 3), constrained_layout=True)

    systems = ["linear", "nonlinear"]
    noises = ["gaussian", "walk"]

    dfs = []
    for system, noise in it.product(systems, noises):
        data = np.load(f"pendulum_{system}_{noise}.npz")
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
            "system": system,
        }))
    df = pd.concat(dfs, ignore_index=True)

    grid = sns.relplot(
        data=df,
        kind="line",
        x="time",
        y=REGRET,
        hue="disturbance",
        style="system",
        height=3.0,
        aspect=1.3,
    )
    # Python's sloppy scoping - `time` was defined in loading loop.
    grid.set(xlim=[time[0], time[-1] + dt])
    grid.savefig("pendulum_costs.pdf")


if __name__ == "__main__":
    main()

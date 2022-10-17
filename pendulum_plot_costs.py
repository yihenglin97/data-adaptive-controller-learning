from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def main():
    names = ["gaussian", "walk"]
    datas = [np.load(f"pendulum_{n}.npz") for n in names]
    colors = ["red", "black"]

    fig, ax = plt.subplots(1, 1, figsize=(5, 3), constrained_layout=True)
    for name, data, color in zip(names, datas, colors):
        dt = data["dt"]
        cost_log = data["cost_log"]
        time = dt * np.arange(len(cost_log))
        cost_cumulative = np.cumsum(cost_log, axis=0).T
        regret = (cost_cumulative[0] - cost_cumulative[1]).squeeze()
        ax.plot(time, regret, label=name, color=color)

    ax.set(
        xlabel="time",
        ylabel="cost(ours) - cost(LQR), cumulative",
        xlim=[time[0], time[-1] + dt],
        #ylim=[-10, 4],  # TODO: get from data
    )
    ax.grid(True)
    ax.legend(loc="lower left", title="disturbance")
    sns.despine(fig)
    fig.savefig("pendulum_costs.pdf")


if __name__ == "__main__":
    main()

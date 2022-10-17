from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def main():
    names = ["gaussian", "walk"]
    datas = [np.load(f"pendulum_{n}.npz") for n in names]

    param_dfs = []
    for data, name in zip(datas, names):
        dt = data["dt"]
        x_log = data["x_log"]
        cost_log = data["cost_log"]
        mass_log = data["mass_log"]
        time = dt * np.arange(len(x_log))
        for controller in ["ours", "LQ"]:
            theta_log = data["theta_log_" + controller]
            for value, paramname in zip(theta_log.T, ["kp", "kd"]):
                param_dfs.append(pd.DataFrame(dict(
                    time=time,
                    gain=value,
                    param=paramname,
                    controller=controller,
                    disturbance=name,
                )))
    param_df = pd.concat(param_dfs, ignore_index=True)

    sns.set_style("ticks", {"axes.grid" : True})
    param_grid = sns.relplot(
        data=param_df,
        kind="line",
        col="disturbance",
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

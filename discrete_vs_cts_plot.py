import os
import multiprocessing

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import solve_discrete_are
import seaborn as sns

from discrete import MPCHorizonSelector
from LinearTracking import LinearTracking
from MPCLTI import MPCLTI


LIGHT_GREY = "#BBBBBB"
ARM = "MPC horizon"


def reverse_category(df, col):
    cat = df[col].astype("category")
    cat = cat.cat.set_categories(cat.cat.categories[::-1], ordered=True)
    df[col] = cat


def main():

    zip = np.load("discrete_vs_cts.npz")
    exp3_batch = zip["exp3_batch"]
    horizon_cost_histories = zip["horizon_cost_histories"]
    dis_cost_history = zip["dis_cost_history"]
    dis_arm_history = zip["dis_arm_history"]
    cts_param_history = zip["cts_param_history"]
    cts_opt_cost_history = zip["cts_opt_cost_history"]
    cts_cost_history = zip["cts_cost_history"]

    # Infer dimensions.
    n_horizons, T = horizon_cost_histories.shape
    horizons = np.arange(n_horizons)
    max_horizon = n_horizons - 1
    n_batches = len(dis_arm_history)

    # Reconstruct some intermediate data.
    step_losses = np.mean(horizon_cost_histories, axis=1)

    plt.rc("figure.constrained_layout", use=True)
    plt.rc("text", usetex=True)
    plt.rc("font", size=12)

    # Plot the mean per-step cost of each MPC horizon with full trust.
    batches = horizon_cost_histories[:, :n_batches*exp3_batch].reshape((n_horizons, n_batches, exp3_batch))
    batch_means = np.mean(batches, axis=-1)
    dfs = []
    for k, means in enumerate(batch_means):
        assert len(means.shape) == 1
        dfs.append(pd.DataFrame({"mean cost": means, ARM: k}))
    df_horizons = pd.concat(dfs, ignore_index=True)
    reverse_category(df_horizons, ARM)
    grid = sns.catplot(
        data=df_horizons,
        kind="violin",
        bw=0.2,
        linewidth=0.5,
        orient="h",
        x="mean cost",
        y=ARM,
        height=3.2,
        aspect=1.1,
        cut=0,
        inner="quartiles",
        color=LIGHT_GREY,
    )
    for ax in grid.axes.flatten():
        ax.grid(True, axis="x")
        for c in ax.collections:
            c.set_edgecolor("black")
    grid.savefig("Plots/batch_sum_hists.pdf")

    # Plot the behavior of EXP3.
    BATCH = "BAPS batch"
    df_exp3 = pd.DataFrame({
        BATCH: np.arange(len(dis_arm_history)),
        ARM: dis_arm_history,
    })
    reverse_category(df_exp3, ARM)
    grid = sns.catplot(
        kind="swarm",
        s=2.5,
        data=df_exp3,
        x=BATCH,
        y=ARM,
        #color="black",
        hue=ARM,
        height=3.2,
        aspect=1.9,
    )
    grid.savefig("Plots/exp3_scatter.pdf")

    # Subsample line plots for less "grittiness".
    skip = T // 1000
    time_skip = np.arange(T)[::skip]

    # Plot the evolution of the policy parameters.
    THETA_I = "$\\theta_i$"
    df_gaps = pd.DataFrame(cts_param_history[::skip])
    df_gaps["time"] = time_skip
    df_gaps = pd.melt(df_gaps, id_vars="time", var_name=THETA_I)
    fig_params = sns.relplot(
        data=df_gaps,
        kind="line",
        x="time",
        y="value",
        hue=THETA_I,
        palette="flare",
        height=2.7,
        aspect=1.3,
    )
    fig_params.savefig("Plots/params_update.pdf")

    # Regret analysis.
    optimal_horizon = np.argmin(step_losses)
    print(f"{optimal_horizon = }")
    dis_opt_cost_history = horizon_cost_histories[optimal_horizon]
    #reg_vs = np.cumsum(dis_cost_history - cts_cost_history)
    reg_dis = np.cumsum(dis_cost_history - dis_opt_cost_history)
    reg_cts = np.cumsum(cts_cost_history - cts_opt_cost_history)
    df_reg = pd.DataFrame({
        "time": time_skip,
        "BAPS vs.\\ optimal": reg_dis[::skip],
        "GAPS vs.\\ final": reg_cts[::skip],
        #"BAPS vs. GAPS": reg_vs[::skip],
    })
    REGRET = "cumulative cost difference"
    df_reg = pd.melt(df_reg, id_vars="time", var_name="algorithm", value_name=REGRET)
    grid = sns.relplot(
        data=df_reg,
        kind="line",
        x="time",
        y=REGRET,
        col="algorithm",
        color="black",
        height=3.0,
        aspect=1.0,
    )
    grid.savefig("Plots/dis_vs_cts_regret.pdf")

    # Show the advantage of using trust values instead of horizon tuning.
    dis_total = np.sum(dis_opt_cost_history)
    cts_total = np.sum(cts_opt_cost_history)
    print(f"LQ cost: optimal discrete = {dis_total:.1f}, optimal continuous = {cts_total:.1f}")
    print(f"         (ratio = {cts_total/dis_total:.2f})")


if __name__ == '__main__':
    main()

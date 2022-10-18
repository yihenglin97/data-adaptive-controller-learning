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

    # Stuff to make our plots look better.
    bar_kwargs = dict(color="#BBBBBB", edgecolor="black", linewidth=0.5)
    bar_axis_set = dict(yticks=horizons, ylim=[-0.5, max_horizon + 0.5], axisbelow=True)
    time_ticks = [0, T // 4, T // 2, T - T // 4, T]

    fig_exp3, (ax_cost, ax_trace, ax_hist) = plt.subplots(
        1, 3,
        figsize=(10, 2.5),
        constrained_layout=True,
        gridspec_kw=dict(
            width_ratios=[1, 2, 1],
        ),
    )

    # Plot the mean per-step cost of each MPC horizon with full trust.
    ax_cost.barh(np.arange(n_horizons), step_losses, **bar_kwargs)
    ax_cost.set(xlabel="mean per-step loss", ylabel="MPC horizon", **bar_axis_set)

    batches = horizon_cost_histories[:, :n_batches*exp3_batch].reshape((n_horizons, n_batches, exp3_batch))
    batch_sums = np.sum(batches, axis=-1)
    dfs = []
    for k, sums in enumerate(batch_sums):
        assert len(sums.shape) == 1
        dfs.append(pd.DataFrame({"cost": sums, "horizon": k}))
    df = pd.concat(dfs, ignore_index=True)
    grid = sns.catplot(
        data=df,
        kind="violin",
        x="horizon",
        y="cost",
    )
    grid.savefig("Plots/batch_sum_hists.pdf")
    print("saved batch sum hists")

    # Plot the behavior of EXP3.
    # Set up histogram bins so we get one centered bin per horizon.
    bins_h = np.concatenate([horizons, [max_horizon + 1]]) - 0.5
    bins_t = np.linspace(0, n_batches, 45)
    ax_trace.hist2d(range(n_batches), dis_arm_history, bins=(bins_t, bins_h), cmap="Greys")
    ax_trace.set(xlabel="EXP3 batch", yticks=horizons)

    counts = np.bincount(dis_arm_history)
    ax_hist.barh(horizons, counts, **bar_kwargs)
    ax_hist.set(xlabel="total batches chosen", **bar_axis_set)
    ax_hist.spines.top.set(visible=False)
    ax_hist.spines.right.set(visible=False)

    for ax in (ax_cost, ax_hist):
        ax.spines.top.set(visible=False)
        ax.spines.right.set(visible=False)
        ax.spines.bottom.set(visible=False)
        ax.grid(axis="x")

    fig_exp3.savefig("Plots/exp3_horizon_selection.pdf")

    # Plot the evolution of the policy parameters.
    cmap = mpl.cm.get_cmap("winter", n_horizons)
    fig_params, ax = plt.subplots(1, 1, figsize=(4.5, 2.5))
    for i in range(max_horizon):
        ax.plot(cts_param_history[:, i], label = f"{i = }", color=cmap(i))
    ax.legend(bbox_to_anchor=(1, 0.5), loc="center left")
    ax.set(xlabel="time", xticks=time_ticks, ylabel="$\\theta_i$ value")
    fig_params.savefig("Plots/params_update.pdf")

    # Regret analysis...
    fig_regret, (ax_dis, ax_cts) = plt.subplots(1, 2, figsize=(8, 2.5))

    # Discrete regret.
    optimal_horizon = np.argmin(step_losses)
    print(f"{optimal_horizon = }")
    dis_opt_cost_history = horizon_cost_histories[optimal_horizon]
    ax_dis.plot(np.cumsum(dis_cost_history - dis_opt_cost_history), color="black", label="discrete")
    ax_dis.legend(loc="lower right")
    ax_dis.set(xlabel="time", xticks=time_ticks, ylabel="regret")

    # Continuous regret.
    ax_cts.plot(np.cumsum(cts_cost_history - cts_opt_cost_history), color="black", label="continuous")
    ax_cts.legend(loc="lower right")
    ax_cts.set(xlabel="time", xticks=time_ticks, ylabel="regret")

    fig_regret.savefig("Plots/dis_vs_cts_regret.pdf")

    # Show the advantage of using trust values instead of horizon tuning.
    dis_total = np.sum(dis_opt_cost_history)
    cts_total = np.sum(cts_opt_cost_history)
    print(f"LQ cost: optimal discrete = {dis_total:.1f}, optimal continuous = {cts_total:.1f}")
    print(f"         (ratio = {cts_total/dis_total:.2f})")


if __name__ == '__main__':
    main()

import matplotlib.pyplot as plt
import numpy as np


def main2():
    npz = np.load("quadrotor_data.npz")
    online = npz["online_cost_history"]
    offline = npz["offline_cost_history"]
    regret = np.cumsum(online - offline)
    fig, ax = plt.subplots(1, 1)
    ax.plot(regret)
    fig.savefig("quadrotor_regret.pdf")


def main():

    npz = np.load("quadrotor_data.npz")

    trip_lengths = npz["online_trip_lengths"]
    pos_history = npz["online_pos_history"]
    pos_desired = npz["online_pos_desired"]
    param_history = npz["online_param_history"]
    param_nominal = npz["online_param_nominal"]

    n_packages = len(trip_lengths)

    # TODO: save/load!!!
    dt = 1.0 / 200
    period = 10.0

    fig_kwargs = dict(figsize=(0.5 * n_packages, 3), constrained_layout=True)
    vline_kwargs = dict(color="black", linewidth=1.0, alpha=0.4, linestyle="--")
    fig_z, ax_z = plt.subplots(1, 1, **fig_kwargs)
    fig_params, ax_params = plt.subplots(1, 1, **fig_kwargs)
    for ax in (ax_params, ax_z):
        changes = dt * np.cumsum(trip_lengths)
        ax.axvline(changes[0], label="new package", **vline_kwargs)
        for t in changes[1:-1]:
            ax.axvline(t, **vline_kwargs)
        ax.grid(axis="y", alpha=0.4)

    T = len(param_history)
    t = dt * np.arange(T)

    #ax.semilogy(-np.array(z_history) - 1)
    ax_z.plot(t[:-1], pos_history[:, 2], label="altitude")
    ax_z.set(xlabel="time (sec)", ylabel="altitude")
    ax_z.legend(loc="upper left")

    fig_xy, ax_xy = plt.subplots(1, 1, figsize=(4, 4), constrained_layout=True)
    #ax.semilogy(-np.array(z_history) - 1)
    end_desired = int(period / dt)
    ax_xy.plot(pos_desired[:end_desired, 0], pos_desired[:end_desired, 1], color="black", linestyle="--", linewidth=1.0, label="desired")
    ax_xy.plot(pos_history[:, 0], pos_history[:, 1], color="black", alpha=0.5, label="actual")
    ax_xy.legend(loc="upper left")
    ax_xy.axis("equal")
    ax_xy.set(xlabel="x", ylabel="y")

    params = np.stack(param_history).T
    params /= param_nominal[:, None]
    hkp, hkv, hkR, hkw = params
    ax_params.axhline(1.0, color="#BBBBBB", linestyle="--")
    plotcmd = ax_params.plot  # Or semilogy
    plotcmd(t, hkp, label="kp")
    plotcmd(t, hkv, label="kv")
    plotcmd(t, hkR, label="kR")
    plotcmd(t, hkw, label="kw")
    ax_params.legend()
    ax_params.set(xlabel="time (sec)", ylabel="param/nominal (ratio)")

    fig_params.savefig("pid_traces.pdf")
    fig_z.savefig("quadrotor_z.pdf")
    fig_xy.savefig("quadrotor_xy.pdf")


if __name__ == "__main__":
    main()
    main2()

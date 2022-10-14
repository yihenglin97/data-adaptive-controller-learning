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

    fig_kwargs = dict(figsize=(n_packages, 4), constrained_layout=True)

    fig_z, ax_z = plt.subplots(1, 1, **fig_kwargs)
    #ax.semilogy(-np.array(z_history) - 1)
    ax_z.plot(pos_history[:, 2])
    ax_z.set(xlabel="step", ylabel="z")

    fig_xy, ax_xy = plt.subplots(1, 1, constrained_layout=True)
    #ax.semilogy(-np.array(z_history) - 1)
    ax_xy.plot(pos_desired[:, 0], pos_desired[:, 1], label="desired")
    ax_xy.plot(pos_history[:, 0], pos_history[:, 1], label="actual")
    ax_xy.legend()
    ax_xy.set(xlabel="x", ylabel="y")

    params = np.stack(param_history).T
    params /= param_nominal[:, None]
    hkp, hkv, hkR, hkw = params
    fig, ax = plt.subplots(1, 1, **fig_kwargs)
    ax.axhline(1.0, color="#BBBBBB", linestyle="--")
    plotcmd = ax.plot  # Or semilogy
    plotcmd(hkp, label="kp")
    plotcmd(hkv, label="kv")
    plotcmd(hkR, label="kR")
    plotcmd(hkw, label="kw")
    ax.set(xlabel="step", ylabel="vs. nominal")

    for ax in (ax, ax_z):
        changes = np.cumsum(trip_lengths)
        ax.axvline(changes[0], label="new package", color="black", linewidth=1.0)
        for t in changes[1:]:
            ax.axvline(t, color="black", linewidth=1.0)
        ax.legend()

    fig.savefig("pid_traces.pdf")
    fig_z.savefig("quadrotor_z.pdf")
    fig_xy.savefig("quadrotor_xy.pdf")


if __name__ == "__main__":
    main()
    main2()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def main():
    data = np.load("pendulum.npz")
    dt = data["dt"]
    x_log = data["x_log"]
    theta_log_LQ = data["theta_log_LQ"]
    theta_log_ours = data["theta_log_ours"]
    cost_log = data["cost_log"]
    mass_log = data["mass_log"]
    time = dt * np.arange(len(x_log))

    theta_log_LQ = np.stack(theta_log_LQ)
    theta_log_ours = np.stack(theta_log_ours)
    fig, (ax_state, ax_param, ax_cost, ax_mass) = plt.subplots(4, 1, figsize=(8, 11), constrained_layout=True)

    ax_state.plot(time, np.degrees(x_log[:, 1, 0]), label="LQR")
    ax_state.plot(time, np.degrees(x_log[:, 0, 0]), label="ours")
    ax_state.legend()
    ax_state.grid()
    ax_state.set(ylabel="angle (degrees)")

    df = pd.concat([
        pd.DataFrame(dict(time=time, value=theta_log_ours[:, 0], param="kp", controller="ours")),
        pd.DataFrame(dict(time=time, value=theta_log_ours[:, 1], param="kd", controller="ours")),
        pd.DataFrame(dict(time=time, value=theta_log_LQ[:, 0], param="kp", controller="LQ")),
        pd.DataFrame(dict(time=time, value=theta_log_LQ[:, 1], param="kd", controller="LQ")),
    ], ignore_index=True)
    sns.lineplot(
        data=df,
        ax=ax_param,
        x="time",
        y="value",
        style="controller",
        hue="param",
    )

    cost_log = np.cumsum(cost_log, axis=0).T
    #ax_cost.plot(time, cost_log[0], label="ours")
    #ax_cost.plot(time, cost_log[1], label="LQR")
    ax_cost.plot(time, cost_log[0] - cost_log[1], label="regret")
    ax_cost.legend()
    ax_cost.set(xlabel="time", ylabel="cost")

    ax_mass.plot(time, mass_log)
    ax_mass.set(xlabel="time", ylabel="mass")

    fig.savefig("pendulum.pdf")


if __name__ == "__main__":
    main()

from core.systems.batched import InvertedPendulum
import torch

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve_continuous_are
import torch
import pandas as pd
import seaborn as sns


_Q = np.eye(2)
_R = 0.1 * np.eye(1)

def _cost(xs):
    xQs = xs @ Q
    return np.sum(xs * xQs, axis=-1)

def pendulum_gains_lqr(m, l):
    g = 9.8
    A = np.array([[0, 1], [g / l, 0]])
    B = np.array([[0], [1 / (m * l**2)]])
    P = solve_continuous_are(A, B, _Q, _R)
    K = np.linalg.solve(_R, B.T @ P)
    kp, kd = K.flat
    return kp, kd


class PDController(torch.nn.Module):
    def __init__(self, k_p, k_d):
        super().__init__()
        self.k_p = k_p
        self.k_d = k_d

    def forward(self, xs):
        thetas, theta_dots = xs.T
        u = -(self.k_p * thetas + self.k_d * theta_dots)
        #u += torch.randn(1)
        u += 0.1
        return u[:, None]


m = 1.0
l = 1.0
kp_opt, kd_opt = pendulum_gains_lqr(m, l)


T = 1000
dt = 0.01

x_log = []
theta_log_LQ = []
theta_log_ours = []

xs = torch.zeros((2, 2))

N = 10
masses = 2 ** np.random.uniform(-1, 1, size=N)

for mass in masses:
    system = InvertedPendulum(m=mass, l=1.0)
    kp_LQ, kd_LQ = pendulum_gains_lqr(mass, l=1.0)
    controller_LQ = PDController(kp_LQ, kd_LQ)
    for i in range(T):
        x_log.append(xs)
        theta_log_LQ.append(np.array([kp_LQ, kd_LQ]))
        theta_log_ours.append(np.array([kp_LQ, kd_LQ]))
        us = torch.concat([
            controller_LQ(xs[0, :][None, :]),
            controller_LQ(xs[1, :][None, :]),
        ], axis=0)
        xs = system.step(xs, us, 0.0, dt)

x_log = np.stack(x_log)
theta_log_LQ = np.stack(theta_log_LQ)
theta_log_ours = np.stack(theta_log_ours)
fig, (ax_state, ax_param) = plt.subplots(2, 1, figsize=(8, 6), constrained_layout=True)

ax_state.plot(x_log[:, 0, 0], label="LQR")
ax_state.plot(x_log[:, 1, 0], label="ours")
ax_state.legend()
ax_state.grid()
ax_state.set(ylabel="angle")

time = dt * np.arange(N * T)
df = pd.concat([
    pd.DataFrame(dict(time=time, val=theta_log_ours[:, 0], param="kd", which="ours")),
    pd.DataFrame(dict(time=time, val=theta_log_ours[:, 1], param="kp", which="ours")),
    pd.DataFrame(dict(time=time, val=theta_log_LQ[:, 0], param="kd", which="LQ")),
    pd.DataFrame(dict(time=time, val=theta_log_LQ[:, 1], param="kp", which="LQ")),
], ignore_index=True)
sns.lineplot(
    data=df,
    ax=ax_param,
    x="time",
    y="val",
    style="which",
    hue="param",
)
#ax_param.plot(theta_log_LQ[:, :], label="LQ")
#ax_param.plot(theta_log_ours[:, :], label="ours")
#ax_param.set(xlabel="step", ylabel="param")
#ax_param.legend(["$k_p$", "$k_d$"])
fig.savefig("pendulum.pdf")

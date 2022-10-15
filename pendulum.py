import numpy as np
from scipy.linalg import solve_continuous_are, solve_discrete_are
import scipy.signal as sig
import torch
import tqdm

from core.systems.batched import InvertedPendulum
from GAPS import GAPSEstimator


class LinearSystem:
    def __init__(self, A, B, dt):
        self.dt = dt
        n, m = B.shape
        sys_c = (A, B, np.eye(n), np.zeros((n, m)))
        Ad, Bd, Cd, Dd, _ = sig.cont2discrete(sys_c, dt)
        assert np.all(Cd.flat == np.eye(n).flat)
        assert np.all(Dd.flat == np.zeros((n, m)).flat)
        self.Ad = torch.tensor(Ad)
        self.Bd = torch.tensor(Bd)

    def step(self, x, u, t0, t1):
        assert t1 - t0 == self.dt
        return x @ self.Ad.T + u @ self.Bd.T


_Q = torch.eye(2)
_R = 0.01 * torch.eye(1)


def _cost(xs, us):
    xQs = xs @ _Q
    uRs = us @ _R
    return torch.sum(xs * xQs, axis=-1) + torch.sum(us * uRs, axis=-1)


def pendulum_linearize(m, l):
    g = 9.8
    A = torch.tensor([[0, 1], [g / l, 0]])
    B = torch.tensor([[0], [1 / (m * l**2)]])
    return A, B


def pendulum_gains_lqrc(m, l):
    A, B = pendulum_linearize(m, l)
    P = solve_continuous_are(A, B, _Q, _R)
    K = np.linalg.solve(_R, B.T @ P)
    kp, kd = K.flat
    return kp, kd


def pendulum_gains_lqrd(m, l, dt):
    A, B = pendulum_linearize(m=m, l=l)
    sys = LinearSystem(A, B, dt)
    Ad, Bd = sys.Ad, sys.Bd
    Qd = dt * _Q
    Rd = dt * _R
    Pd = solve_discrete_are(Ad, Bd, Qd, Rd)
    Kd = np.linalg.solve(Rd + Bd.T @ Pd @ Bd, Bd.T @ Pd @ Ad)
    kp, kd = Kd.flat
    return kp, kd


class PDController(torch.nn.Module):
    def __init__(self, k_p, k_d):
        super().__init__()
        self.k_p = k_p
        self.k_d = k_d

    def forward(self, xs):
        thetas, theta_dots = xs.T
        u = -(self.k_p * thetas + self.k_d * theta_dots)
        return u[:, None]


def controller_ours(x, theta):
    return PDController(*theta)(x[None, :])[0]


def t2np(t):
    if t is not None:
        return t.detach().numpy()
    return None


def ulprocess(seed, noise, attraction):
    """Ornstein-Uhlenbeck random walk generator.

    When attraction == 1, degenerates into Gaussian noise.
    """
    npr = np.random.default_rng(seed=seed)
    if attraction > 1:
        raise ValueError("attraction should be <= 1.")
    x = 0.0
    while True:
        x += noise * npr.normal() - attraction * x
        yield x


def main():
    dt = 0.01 # Discretization time interval.
    N = 10    # Number of step-changes in mass.
    T = 10000  # Number of timesteps per step-change in mass.

    buf_len = int(1.0 / dt)
    rate = 1e1
    estimator = GAPSEstimator(buffer_length=buf_len)
    theta = torch.tensor(pendulum_gains_lqrd(1.0, 1.0, dt))
    prev_dgdx = None
    prev_dgdu = None

    np.random.seed(100)
    masses = 2 ** np.random.uniform(-1, 1, size=N)
    disturbance = ulprocess(0, 0.2 * dt, 0.2)
    xs = torch.zeros((2, 2))

    x_log = []
    mass_log = []
    theta_log_LQ = []
    theta_log_ours = []
    cost_log = []

    for mass in tqdm.tqdm(masses):
        system = InvertedPendulum(m=mass, l=1.0)
        #system = LinearSystem(*pendulum_linearize(m=mass, l=1.0), dt)
        def dynamics(x, u):
            return system.step(x[None, :], u[None, :], 0.0, dt)[0]

        kp_LQ, kd_LQ = pendulum_gains_lqrd(mass, l=1.0, dt=dt)
        controller_LQ = PDController(kp_LQ, kd_LQ)

        for i in range(T):
            # Get actions.
            us = torch.concat([
                controller_ours(xs[0], theta)[None, :],
                controller_LQ(xs[1][None, :]),
            ], axis=0)

            # Log everything.
            x_log.append(xs)
            theta_log_LQ.append(np.array([kp_LQ, kd_LQ]))
            theta_log_ours.append(theta)
            cost_log.append(_cost(xs, us))
            mass_log.append(mass)

            # Get controller derivatives.
            dudx, dudtheta = torch.autograd.functional.jacobian(
                controller_ours,
                (xs[0], theta),
                vectorize=True,
            )
            estimator.add_partial_u(dudx.detach().numpy(), dudtheta.detach().numpy())

            # Get system derivatives.
            dgdx, dgdu = torch.autograd.functional.jacobian(dynamics, (xs[0], us[0]), vectorize=True)
            dfdx, dfdu = torch.autograd.functional.jacobian(_cost, (xs[0], us[0]), vectorize=True)
            # Gradient sanity check.
            assert np.dot(dfdx, xs[0]) >= 0
            assert np.dot(dfdu, us[0]) >= 0
            derivatives = (dfdx, dfdu, prev_dgdx, prev_dgdu)
            prev_dgdx = dgdx
            prev_dgdu = dgdu

            # Gradient step.
            G = estimator.update(*map(t2np, derivatives))
            theta = theta - rate * G

            # Dynamics step.
            xs = system.step(xs, us, 0.0, dt)
            xs[:, 1] += next(disturbance)

    # Save data.
    x_log = np.stack(x_log)
    np.savez(
        "pendulum.npz",
        dt=dt,
        x_log=x_log,
        theta_log_LQ=np.stack(theta_log_LQ),
        theta_log_ours=np.stack(theta_log_ours),
        cost_log=np.stack(cost_log),
        mass_log=np.array(mass_log),
    )


if __name__ == "__main__":
    main()

import matplotlib.pyplot as plt
import numpy as np
import torch

from GAPS import GAPSEstimator
from torchenv import TorchEnv


_GRAV = torch.Tensor([0, 0, -9.81])


def _hat(w):
    """Returns matrix A such that Av = w x v for all v."""
    x, y, z = w
    return torch.tensor([
        [ 0, -z,  y],
        [ z,  0, -x],
        [-y,  x,  0]
    ])


def _vee(W):
    """Inverse of _hat(w)."""
    return torch.tensor([W[2, 1], W[0, 2], W[1, 0]])


_ARM = 0.315  # meters
_TORQUE = 8.004  # ???
_FORCES_TO_THRUST_MOMENT = torch.tensor([
    [       1,       1,        1,       1],
    [       0,   -_ARM,        0,    _ARM],
    [    _ARM,       0,    -_ARM,       0],
    [-_TORQUE, _TORQUE, -_TORQUE, _TORQUE],
], dtype=torch.float64)
_THRUST_MOMENT_TO_FORCES = torch.linalg.inv(_FORCES_TO_THRUST_MOMENT)


nominal_mass = 4.34  # kilograms
nominal_inertia = np.array([0.0820, 0.0845, 0.1377])  # kg * m^2


def unpack(x):
    p = x[:3]
    v = x[3:6]
    R = x[6:15].reshape((3, 3))
    w = x[15:]
    return p, v, R, w


def pack(p, v, R, w):
    return torch.cat([p, v, R, w])


def make_quadrotor_dynamics(mass, inertia, dt):
    inertia = torch.tensor(inertia)
    def quadrotor_dynamics(x, u):
        assert len(x) == 18
        p, v, R, w = unpack(x)

        thrust_moment = _FORCES_TO_THRUST_MOMENT @ u
        thrust = thrust_moment[0]
        moment = thrust_moment[1:]

        dp = v
        dv = _GRAV + (thrust / mass) * R[:, 2]
        dR = R @ _hat(w)
        dw = torch.diag(1.0 / inertia) @ (moment - torch.linalg.cross(w, torch.diag(inertia) @ w))

        return torch.cat([
            p + dt * dp,
            v + dt * dv,
            (R + dt * dR).flatten(),
            w + dt * dw,
        ])
    return quadrotor_dynamics

def make_quadrotor_control(mass, inertia):
    inertia = torch.tensor(inertia)
    def quadrotor_control(x, theta):
        kp, kv, kR, kw = theta
        p, v, R, w = unpack(x)
        eR = 0.5 * _vee(R - R.T)
        force = -(
            kp * p
            + kv * v
            + mass * _GRAV
        )
        thrust = torch.dot(force, R[:, 2])
        moment = (
            -kR * eR
            - kw * w
            # Inertia is diagonal, so matrix mult = elementwise mult
            + torch.linalg.cross(w, inertia * w)
            # Simplified when desired angular velocity is zero.
        )
        return _THRUST_MOMENT_TO_FORCES @ torch.cat([thrust[None], moment])
    return quadrotor_control


_TRIANGLE_INEQUALITY_CHECK = np.array([
    [ 1,  1, -1],
    [ 1, -1,  1],
    [-1,  1,  1]
])


def _random_inertia(npr, scale):
    while True:
        inertia = scale ** npr.uniform(-1, 1, size=3)
        if np.all(_TRIANGLE_INEQUALITY_CHECK @ inertia >= 0):
            return inertia


_IDENTITY_STATE = np.concatenate([
    np.zeros(3),
    np.zeros(3),
    np.eye(3).flat,
    np.zeros(3),
])


def main():
    npr = np.random.default_rng(seed=0)
    T = 20000
    n_packages = 20
    trip_lengths = 2 ** npr.uniform(-1, 1, size=n_packages)
    trip_lengths *= T / np.sum(trip_lengths)
    trip_lengths = trip_lengths.astype(np.int)

    inertia_scales = np.stack([_random_inertia(npr, 2.0) for _ in range(n_packages)])
    inertia_scales[0] = np.ones(3)
    inertias = inertia_scales * nominal_inertia[None, :]

    mass_scales = 2 ** npr.uniform(-1, 1, size=n_packages)
    mass_scales[0] = 1.0
    masses = mass_scales * nominal_mass

    dt = 1.0 / 50

    estimator = GAPSEstimator(buffer_length=100)

    dynamics = make_quadrotor_dynamics(masses[0], inertias[0], dt)
    def cost(x, u):
        # TODO: per-state costs
        p, v, R, w = unpack(x)
        return (
            torch.sum(p[:3] ** 2)
            + 0.01 * torch.sum(v[:3] ** 2)
            + 0.0001 * torch.sum(u ** 2)
        )

    env = TorchEnv(dynamics, cost, _IDENTITY_STATE)

    param_nominal = np.array([16.0, 5.6, 8.81, 2.54])
    param = param_nominal.copy()
    param_history = [param]
    eta = 1e0

    z_history = []

    for i in range(n_packages):
        mass = masses[i]
        inertia = inertias[i]
        dynamics = make_quadrotor_dynamics(mass, inertia, dt)
        env.change_dynamics(dynamics)
        controller = make_quadrotor_control(nominal_mass, nominal_inertia)
        for t in range(trip_lengths[i]):
            x = env.observe()
            z_history.append(float(x[2]))
            u = controller(torch.tensor(x), torch.tensor(param))
            dudx, dudtheta = torch.autograd.functional.jacobian(
                controller,
                (torch.tensor(x), torch.tensor(param)),
                vectorize=True,
            )
            dudx = dudx.detach().numpy()
            dudtheta = dudtheta.detach().numpy()
            estimator.add_partial_u(dudx, dudtheta)
            print(f"{u = }")
            derivatives = env.step(u)
            G = estimator.update(*derivatives)
            print(f"{G = }")
            # TODO: projection!
            param = param - eta * G
            param_history.append(param)

    fig_kwargs = dict(figsize=(12, 4), constrained_layout=True)
    fig_z, ax_z = plt.subplots(1, 1, **fig_kwargs)
    #ax.semilogy(-np.array(z_history) - 1)
    ax_z.plot(z_history)
    ax_z.set(xlabel="step", ylabel="z")

    params = np.stack(param_history).T
    params /= param_nominal[:, None]
    hkp, hkv, hkR, hkw = params
    fig, ax = plt.subplots(1, 1, **fig_kwargs)
    ax.plot(hkp, label="kp")
    ax.plot(hkv, label="kv")
    ax.plot(hkR, label="kR")
    ax.plot(hkw, label="kw")
    ax.set(xlabel="step", ylabel="vs. nominal")

    for ax in (ax, ax_z):
        changes = np.cumsum(trip_lengths)
        ax.axvline(changes[0], label="new package", color="black", linewidth=1.0)
        for t in changes[1:]:
            ax.axvline(t, color="black", linewidth=1.0)
        ax.legend()

    fig.savefig("pid_traces.pdf")
    fig_z.savefig("quadrotor_z.pdf")


if __name__ == "__main__":
    main()

import matplotlib.pyplot as plt
import numpy as np
import torch

from GAPS import GAPSEstimator
from torchenv import TorchEnv

# z points down

_GRAV = torch.Tensor([0, 0, 9.81])


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


def make_quadrotor_dynamics(mass, inertia, dt):
    inertia = torch.tensor(inertia)
    def quadrotor_dynamics(x, u):
        assert len(x) == 18
        p, v, R, w = unpack(x)

        thrust_moment = _FORCES_TO_THRUST_MOMENT @ u
        thrust = thrust_moment[0]
        moment = thrust_moment[1:]

        dp = v
        dv = _GRAV - (thrust / mass) * R[:, 2]
        dw = torch.diag(1.0 / inertia) @ (moment - torch.linalg.cross(w, torch.diag(inertia) @ w))

        # Rodrigues
        if False:
            wnorm = torch.linalg.norm(w)
            if wnorm > 1e-6:
                #w2 = R @ w
                #K = _hat(w2 / wnorm)
                K = R @ _hat(w / wnorm)
                angle = dt * wnorm
                Rstep = torch.eye(3) + torch.sin(angle) * K + (1.0 - torch.cos(angle)) * K @ K
                Rnext = Rstep @ R
            else:
                Rnext = R
        else:
            dR = R @ _hat(w)
            Rnext = R + dt * dR

        return torch.cat([
            p + dt * dp,
            v + dt * dv,
            Rnext.flatten(),
            w + dt * dw,
        ])
    return quadrotor_dynamics


def _normalize(x):
    return x / torch.linalg.norm(x)


def make_quadrotor_control(mass, inertia):
    inertia = torch.tensor(inertia, requires_grad=False)
    def quadrotor_control(x, pdes, vdes, theta):
        kp, kv, kR, kw = theta
        p, v, R, w = unpack(x)
        # We set feedforward linear and angular accelerations to 0 - let the
        # controller handle it.
        ep = p - pdes
        ev = v - vdes
        forcevec = -(
            kp * ep
            + kv * ev
            + mass * _GRAV
        )
        R3 = -_normalize(forcevec)
        R2 = _normalize(torch.linalg.cross(R3, torch.tensor([1, 0, 0], dtype=torch.double)))
        R1 = torch.linalg.cross(R2, R3)
        assert np.isclose(torch.linalg.norm(R1).detach().numpy(), 1.0, atol=1e-6)
        Rd = torch.column_stack([R1, R2, R3])
        wd = torch.zeros(3, dtype=torch.double)  # TODO: correct?

        eR = 0.5 * _vee(Rd.T @ R - R.T @ Rd)
        ew = w - R.T @ (Rd @ wd)
        thrust = -torch.dot(forcevec, R[:, 2])
        moment = (
            - kR * eR
            - kw * ew
            # Inertia is diagonal, so matrix mult = elementwise mult
            + torch.linalg.cross(w, inertia * w)
            - inertia * torch.linalg.cross(w, R.T @ (Rd @ wd))
        )
        return _THRUST_MOMENT_TO_FORCES @ torch.cat([thrust[None], moment])
    return quadrotor_control


def make_quadrotor_cost(pdes, vdes):
    def cost(x, u):
        p, v, R, w = unpack(x)
        # TODO: R, w error costs?
        return (
            torch.sum((p - pdes) ** 2)
            #+ 0.01 * torch.sum((v - vdes) ** 2)
            + 0.0001 * torch.sum(u ** 2)
        )
    return cost


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
    n_packages = 20
    T = n_packages * 1000
    trip_lengths = 2 ** npr.uniform(-1, 1, size=n_packages)
    trip_lengths *= T / np.sum(trip_lengths)
    trip_lengths = trip_lengths.astype(int)

    mass_scales = 2 ** npr.uniform(-1, 1, size=n_packages)
    mass_scales[0] = 1.0
    masses = mass_scales * nominal_mass

    inertia_scales = np.stack([_random_inertia(npr, 2.0) for _ in range(n_packages)])
    inertia_scales[0] = np.ones(3)
    inertias = mass_scales[:, None] * inertia_scales * nominal_inertia[None, :]

    dt = 1.0 / 200

    # Compute desired trajectory.
    circle_period = 10.0 # seconds
    circle_radius = 1.0
    omega = 2.0 * np.pi / circle_period
    t = dt * np.arange(T)
    pdes = circle_radius * np.stack([np.cos(omega * t) - 1, np.sin(omega * t), np.zeros(T)]).T
    vdes = circle_radius * np.stack([-omega * np.sin(omega * t), omega * np.cos(omega * t), np.zeros(T)]).T

    estimator = GAPSEstimator(buffer_length=1000)

    dynamics = make_quadrotor_dynamics(masses[0], inertias[0], dt)
    cost = make_quadrotor_cost(pdes=np.zeros(3), vdes=np.zeros(3))

    env = TorchEnv(dynamics, cost, _IDENTITY_STATE)

    param_nominal = np.array([16.0, 5.6, 8.81, 2.54])
    param = param_nominal.copy()
    param_history = [param]
    eta = 1e-1

    p_history = []

    itot = 0
    for i in range(n_packages):
        mass = masses[i]
        inertia = inertias[i]
        dynamics = make_quadrotor_dynamics(mass, inertia, dt)
        env.change_dynamics(dynamics)
        controller = make_quadrotor_control(nominal_mass, nominal_inertia)
        for t in range(trip_lengths[i]):
            x = env.observe()
            p, v, R, w = unpack(x)
            U, E, VT = np.linalg.svd(R)
            R = U @ VT
            x = np.concatenate([p, v, R.flat, w])

            p_history.append(x[:3])
            pdesi = torch.tensor(pdes[itot], requires_grad=False)
            vdesi = torch.tensor(vdes[itot], requires_grad=False)
            cost = make_quadrotor_cost(pdesi, vdesi)
            env.change_cost(cost)
            u = controller(torch.tensor(x), pdesi, vdesi, torch.tensor(param))
            dudx, _, _, dudtheta = torch.autograd.functional.jacobian(
                controller,
                (torch.tensor(x), pdesi, vdesi, torch.tensor(param)),
                vectorize=True,
            )
            dudx = dudx.detach().numpy()
            dudtheta = dudtheta.detach().numpy()
            estimator.add_partial_u(dudx, dudtheta)
            derivatives = env.step(u)

            # Gradient step
            G = estimator.update(*derivatives)
            param = param - eta * G

            # Projection
            param[param < 0] = 0
            param = np.minimum(param, 10 * param_nominal)

            param_history.append(param)
            itot += 1

    p_history = np.stack(p_history)

    fig_kwargs = dict(figsize=(12, 4), constrained_layout=True)

    fig_z, ax_z = plt.subplots(1, 1, **fig_kwargs)
    #ax.semilogy(-np.array(z_history) - 1)
    ax_z.plot(p_history[:, 2])
    ax_z.set(xlabel="step", ylabel="z")

    fig_xy, ax_xy = plt.subplots(1, 1, constrained_layout=True)
    #ax.semilogy(-np.array(z_history) - 1)
    ax_xy.plot(pdes[:, 0], pdes[:, 1], label="desired")
    ax_xy.plot(p_history[:, 0], p_history[:, 1], label="actual")
    ax_xy.legend()
    ax_xy.set(xlabel="x", ylabel="y")

    params = np.stack(param_history).T
    params /= param_nominal[:, None]
    hkp, hkv, hkR, hkw = params
    fig, ax = plt.subplots(1, 1, **fig_kwargs)
    ax.semilogy(hkp, label="kp")
    ax.semilogy(hkv, label="kv")
    ax.semilogy(hkR, label="kR")
    ax.semilogy(hkw, label="kw")
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

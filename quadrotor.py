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
        dv -= 0.02 * v  # Damping.
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
        forces = _THRUST_MOMENT_TO_FORCES @ torch.cat([thrust[None], moment])
        forces[forces < 0] = 0
        return forces
    return quadrotor_control


def make_quadrotor_cost(pdes, vdes):
    def cost(x, u):
        p, v, R, w = unpack(x)
        # TODO: R, w error costs?
        return (
            torch.sum((p - pdes) ** 2)
            #+ 0.01 * torch.sum((v - vdes) ** 2)
            + 1e-2 * torch.sum(u ** 2)
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

def run_fixed(dt, trip_lengths, masses, inertias, pdes, vdes, param):
    n_packages = len(trip_lengths)
    npr = np.random.default_rng(seed=0)

    dynamics = make_quadrotor_dynamics(masses[0], inertias[0], dt)
    cost = make_quadrotor_cost(pdes=np.zeros(3), vdes=np.zeros(3))

    p_history = []
    cost_history = []
    x = torch.tensor(_IDENTITY_STATE)

    itot = 0
    for i in range(n_packages):
        mass = masses[i]
        inertia = inertias[i]
        dynamics = make_quadrotor_dynamics(mass, inertia, dt)

        mass_estimate = mass * (1.2 ** npr.uniform(-1, 1))
        inertia_estimate = (mass_estimate / nominal_mass) * nominal_inertia
        controller = make_quadrotor_control(mass_estimate, inertia_estimate)

        for t in range(trip_lengths[i]):
            p, v, R, w = unpack(x)
            U, E, VT = np.linalg.svd(R)
            R = U @ VT
            w += dt * 0.1 * np.random.normal(size=3)
            x = np.concatenate([p, v, R.flat, w])

            pdesi = torch.tensor(pdes[itot], requires_grad=False)
            vdesi = torch.tensor(vdes[itot], requires_grad=False)
            cost = make_quadrotor_cost(pdesi, vdesi)

            p_history.append(x[:3])

            u = controller(torch.tensor(x), pdesi, vdesi, torch.tensor(param))
            x = dynamics(torch.tensor(x), u).detach().numpy()

            cost_history.append(cost(torch.tensor(x), u).detach().numpy())

            itot += 1

    cost_history = np.stack(cost_history).squeeze()
    p_history = np.stack(p_history)

    return dict(
        trip_lengths=trip_lengths,
        pos_desired=pdes,
        pos_history=p_history,
        param_nominal=param,
        cost_history=cost_history
    )


def run_online(dt, trip_lengths, masses, inertias, pdes, vdes):

    npr = np.random.default_rng(seed=0)
    n_packages = len(trip_lengths)
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

        mass_estimate = mass * (1.2 ** npr.uniform(-1, 1))
        inertia_estimate = (mass_estimate / nominal_mass) * nominal_inertia
        controller = make_quadrotor_control(mass_estimate, inertia_estimate)
        for t in range(trip_lengths[i]):
            x = env.observe()
            p, v, R, w = unpack(x)
            U, E, VT = np.linalg.svd(R)
            R = U @ VT
            w += dt * 0.1 * np.random.normal(size=3)
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
            print(f"{G = }")

            # Projection
            param[param < 0] = 0
            param = np.minimum(param, 10 * param_nominal)

            param_history.append(param)
            itot += 1

    cost_history = np.array(env.cost_history)
    p_history = np.stack(p_history)

    return dict(
        trip_lengths=trip_lengths,
        pos_desired=pdes,
        pos_history=p_history,
        param_history=param_history,
        param_nominal=param_nominal,
        cost_history=cost_history
    )

def dict_key_map(d, f):
    return {f(k) : v for k, v in d.items()}

def dict_key_prepend(d, s):
    return dict_key_map(d, lambda k: s + k)

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

    inertia_scales = np.stack([_random_inertia(npr, 4.0) for _ in range(n_packages)])
    inertia_scales[0] = np.ones(3)
    inertias = mass_scales[:, None] * inertia_scales * nominal_inertia[None, :]

    dt = 1.0 / 200

    # Compute desired trajectory.
    circle_period = 10.0 # seconds
    circle_radius = 2.0
    omega = 2.0 * np.pi / circle_period
    t = dt * np.arange(T)
    pdes = circle_radius * np.stack([np.cos(omega * t) - 1, np.sin(omega * t), np.zeros(T)]).T
    vdes = circle_radius * np.stack([-omega * np.sin(omega * t), omega * np.cos(omega * t), np.zeros(T)]).T

    online_result = run_online(dt, trip_lengths, masses, inertias, pdes, vdes)

    last_param = online_result["param_history"][-1]
    offline_result = run_fixed(dt, trip_lengths, masses, inertias, pdes, vdes, last_param)

    online_cost = np.sum(online_result["cost_history"])
    offline_cost = np.sum(offline_result["cost_history"])
    print(f"{online_cost = }, {offline_cost = }")

    all_results = {
        **dict_key_prepend(online_result, "online_"),
        **dict_key_prepend(offline_result, "offline_"),
    }
    np.savez("quadrotor_data.npz", **all_results)


if __name__ == "__main__":
    main()

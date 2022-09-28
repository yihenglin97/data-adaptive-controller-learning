import LinearTracking
import MPCLTI
import matplotlib.pyplot as plt
from tqdm import trange
import numpy as np
from scipy import linalg as la

if __name__ == '__main__':
    k = 7
    n = 4
    m = 2
    buffer_length = 50
    dt = 0.2
    A, B = np.eye(4), np.zeros((4, 2))
    A[0, 2], A[1, 3] = dt, dt
    B[2, 0], B[3, 1] = dt, dt
    Q = np.eye(4)
    Q[2, 2], Q[3, 3] = 0, 0
    R = np.eye(2) * 0.1
    P = la.solve_discrete_are(A, B, Q, R)
    initial_param = np.zeros(k)
    MPC_instance = MPCLTI.MPCLTI(initial_param=initial_param, buffer_length=buffer_length, learning_rate=0.1)

    x_0 = [3, 0, 0, 0]
    T = 3000

    # Generate a trajectory for tracking
    V = np.zeros((n, T + 1))
    V[:, 0] = x_0
    for t in range(1, T + 1):
        V[0, t] = 2.0 * np.cos(np.pi * t * dt / 30) + np.cos(np.pi * t * dt / 5)
        V[1, t] = 2.0 * np.sin(np.pi * t * dt / 30) + np.sin(np.pi * t * dt / 5)
        V[2, t] = (V[0, t] - V[0, t - 1]) / dt
        V[3, t] = (V[1, t] - V[1, t - 1]) / dt
    plt.plot(V[0, :], V[1, :], label="Target")
    plt.savefig("Plots/Trajectory_to_track.jpg")

    LTI_instance = LinearTracking.LinearTracking(A = A, B = B, Q = Q, R = R, Qf = P, init_state = x_0, traj = V, w_scale=0.1, e_scale=0)

    for t in range(T):
        current_state, context = LTI_instance.observe(k)
        control_action = MPC_instance.decide_action(current_state, context)
        grad_tuple = LTI_instance.step(control_action)
        MPC_instance.update_param(grad_tuple)

    total_cost, whole_trajectory = LTI_instance.reset()
    print(total_cost)
    plt.plot(whole_trajectory[0, :], whole_trajectory[1, :], label="Controller")
    plt.legend()
    plt.savefig("Plots/Trajectory_of_controller.jpg")

    param_history = MPC_instance.param_history
    param_history = np.transpose(np.stack(param_history, axis=0))
    plt.clf()
    for i in range(k):
        plt.plot(param_history[i, :], label = "i = {}".format(i))
    plt.legend()
    plt.savefig("Plots/Params_update.jpg")


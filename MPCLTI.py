import Controller
from scipy import linalg as la
import numpy as np
import cvxpy as cp

class MPCLTI(Controller.Controller):
    def __init__(self, initial_param, buffer_length, learning_rate):
        super().__init__(initial_param, buffer_length, learning_rate)
        self.partial_u_theta = []
        self.partial_u_x = []
        self.param_history = [initial_param]
        self.A = None
        self.B = None
        self.Q = None
        self.R = None
        self.P = None
        self.K = None
        self.Multipliers = []
        self.n = None
        self.m = None
        self.max_k = initial_param.shape[0]

    def decide_action(self, state, context):
        is_terminal, predicted_disturbances, V, sys_params = context
        if self.P is None:
            self.A, self.B, self.Q, self.R = sys_params
            self.P = la.solve_discrete_are(self.A, self.B, self.Q, self.R)
            H = np.linalg.inv(self.R + np.transpose(self.B)@self.P@self.B)@np.transpose(self.B)
            self.K = H@self.P@self.A
            F = self.A - self.B@self.K
            temp_mat = - self.P
            for i in range(self.max_k):
                self.Multipliers.append(H@temp_mat)
                temp_mat = np.transpose(F)@temp_mat
            self.n = self.B.shape[0]
            self.m = self.B.shape[1]
        k = predicted_disturbances.shape[1]
        control_action = - self.K@(state - V[:, 0])
        grad_list = []
        for i in range(k):
            grad_list.append(self.Multipliers[i]@(predicted_disturbances[:, i]))
            control_action += (self.param[i]*grad_list[-1] + self.Multipliers[i]@(self.A@V[:, i] - V[:, i+1]))
        for i in range(self.max_k - k):
            grad_list.append(np.zeros(self.m))
        self.partial_u_x.append(-self.K)
        self.partial_u_theta.append(np.transpose(np.stack(grad_list, axis=0)))
        return control_action

    def update_param(self, grads):
        partial_f_x, partial_f_u, partial_g_x, partial_g_u = grads
        new_buffer = []
        G = None
        if len(self.buffer) == 0:
            G = partial_f_u @ self.partial_u_theta[-1]
            # update the buffer
            new_buffer = [self.partial_u_theta[-1]]
        else:
            grad_sum = partial_g_u @ self.buffer[0]
            current_buffer_len = len(self.buffer)
            new_buffer = [self.partial_u_theta[-1], partial_g_u @ self.buffer[0]]
            for b in range(1, current_buffer_len):
                new_buffer.append((partial_g_x + partial_g_u @ self.partial_u_x[-2]) @ self.buffer[b])
                grad_sum += new_buffer[-1]
            G = (partial_f_x + partial_f_u @ self.partial_u_x[-1])@grad_sum + partial_f_u @ self.partial_u_theta[-1]

        new_param = self.param - self.learning_rate * G
        # the projection step
        for j in range(self.max_k):
            if new_param[j] > 1.0:
                new_param[j] = 1.0
            elif new_param[j] < 0:
                new_param[j] = 0.0
        self.param = new_param
        self.param_history.append(new_param)
        if len(new_buffer) > self.buffer_length:
            self.buffer = new_buffer[:self.buffer_length]
        else:
            self.buffer = new_buffer
        return self.param


if __name__ == "__main__":
    k = 7
    n = 4
    m = 2
    dt = 0.1
    A, B = np.eye(4), np.zeros((4, 2))
    A[0, 2], A[1, 3] = dt, dt
    B[2, 0], B[3, 1] = dt, dt
    Q = np.eye(4)
    R = np.eye(2) * 0.1
    P = la.solve_discrete_are(A, B, Q, R)
    initial_param = np.ones(k)
    MPC_instance = MPCLTI(initial_param=initial_param, learning_rate=0.1)

    state = np.ones(n)
    is_terminal = False
    predicted_disturbances = np.zeros((n, k))
    V = np.zeros((n, k+1))
    sys_params = (A, B, Q, R)
    context = (is_terminal, predicted_disturbances, V, sys_params)

    MPC_instance.decide_action(state, context)





"""
x = cp.Variable((n, k + 1))
        u = cp.Variable((m, k))
        theta = cp.Parameter(k)
        current_state = cp.Parameter(n)
        cost = 0
        constr = []
        for t in range(k):
            cost += cp.quad_form(x[:, t] - V[:, t], self.Q) + cp.quad_form(u[:, t], self.R)
            constr += [x[:, t + 1] == A @ x[:, t] + B @ u[:, t] + theta[t] * predicted_disturbances[:, t]]
        constr += [x[:, 0] == current_state]
        cost += cp.quad_form(x[:, k] - V[:, k], self.P)
        problem = cp.Problem(cp.Minimize(cost), constr)

        theta.value = self.param
        current_state.value = state

        problem.solve(solver=cp.SCS, requires_grad=True, eps=1e-10)
        x.gradient = np.zeros((n, k+1))
        u.gradient = np.zeros((m, k))
        u.gradient[1, 0] = 1
        problem.backward()

        print("target: ", - self.K)
        print("gradient: ", current_state.gradient)
        print("shape: ", current_state.gradient.shape)
        print("u target: ", - self.K @ state)
        print("u: ", u.value[:, 0])
"""
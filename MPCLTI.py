from numba import njit
import numpy as np
from scipy import linalg as la

import Controller


@njit
def _step(multipliers, multipliersA, max_k, param, control_action, predicted_disturbances, V):
    param_dim = multipliers[0].shape[0]
    grads = np.zeros((param_dim, max_k))
    k = predicted_disturbances.shape[0]
    for i in range(k):
        grads[:, i] = multipliers[i] @ predicted_disturbances[i]
        control_action += param[i]*grads[:, i] + multipliersA[i]@(V[i] - V[i+1])
    return grads

class MPCLTI(Controller.Controller):
    def __init__(self, initial_param, buffer_length, learning_rate, horizon=None):
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
        self.MultipliersA = []  # Performance optimization
        self.n = None
        self.m = None
        if horizon is None:
            self.max_k = initial_param.shape[0]
        else:
            assert initial_param.size == 1
            self.max_k = horizon

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
                self.MultipliersA.append(H@temp_mat@self.A)
                temp_mat = np.transpose(F)@temp_mat
            if self.max_k > 0:
                self.Multipliers = np.stack(self.Multipliers)
                self.MultipliersA = np.stack(self.MultipliersA)
            self.n = self.B.shape[0]
            self.m = self.B.shape[1]
        k = predicted_disturbances.shape[1]
        control_action = - self.K@(state - V[:, 0])
        self.partial_u_x.append(-self.K)
        if self.param.size > 1:
            # Copy to make contiguous.
            grads = _step(self.Multipliers, self.MultipliersA, self.max_k, self.param, control_action, predicted_disturbances.T.copy(), V.T.copy())
            # Mutates control_action
            self.partial_u_theta.append(grads)
        else:
            grad = np.zeros_like(control_action)
            for i in range(k):
                this_grad = self.Multipliers[i]@(predicted_disturbances[:, i])
                control_action += (self.param*this_grad + self.Multipliers[i]@(self.A@V[:, i] - V[:, i+1]))
                grad += this_grad
            self.partial_u_theta.append(grad)

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
            premul = partial_g_x + partial_g_u @ self.partial_u_x[-2]
            if current_buffer_len > 1:
                Brest = self.buffer[1:]
                new = premul @ Brest
                grad_sum += np.sum(new, axis=0)
                new_buffer.extend(list(new))
            G = (partial_f_x + partial_f_u @ self.partial_u_x[-1])@grad_sum + partial_f_u @ self.partial_u_theta[-1]

        new_param = self.param - self.learning_rate * G
        # the projection step
        new_param = np.clip(new_param, 0.0, 1.0)
        self.param = new_param
        self.param_history.append(new_param)
        if len(new_buffer) > self.buffer_length:
            self.buffer = new_buffer[:self.buffer_length]
        else:
            self.buffer = new_buffer
        return self.param

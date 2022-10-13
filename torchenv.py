import torch


class TorchEnv:
    def __init__(self, dynamics, cost, init_state):
        self.dynamics = dynamics
        self.cost = cost
        self.x = torch.tensor(init_state)

        self.partial_f_x = []   # partial f_t/x_t
        self.partial_f_u = []   # partial f_t/u_t
        self.partial_g_x = []   # partial g_t/x_t
        self.partial_g_u = []   # partial g_t/u_t
        self.x_history = [self.x.detach().numpy()]
        self.cost_history = []

    def change_dynamics(self, dynamics):
        self.dynamics = dynamics

    def change_cost(self, cost):
        self.cost = cost

    def observe(self):
        return self.x.detach().numpy()

    # return the available gradients (f_t/x_t, f_t/u_t, g_t/x_t, g_t/u_t)
    def step(self, u):
        stage_cost = float(self.cost(self.x, u).detach())
        self.cost_history.append(stage_cost)

        dgdx, dgdu = torch.autograd.functional.jacobian(self.dynamics, (self.x, u), vectorize=True)
        dfdx, dfdu = torch.autograd.functional.jacobian(self.cost, (self.x, u), vectorize=True)
        dgdx = dgdx.detach().numpy()
        dgdu = dgdu.detach().numpy()
        dfdx = dfdx.detach().numpy()
        dfdu = dfdu.detach().numpy()
        self.partial_f_x.append(dfdx)
        self.partial_f_u.append(dfdu)
        self.partial_g_x.append(dgdx)
        self.partial_g_u.append(dgdu)

        self.x = self.dynamics(self.x, u).detach()
        self.x_history.append(self.x.numpy())

        if len(self.partial_g_x) < 2:
            grad_tuple = (dfdx, dfdu, None, None)
        else:
            grad_tuple = (dfdx, dfdu, self.partial_g_x[-2], self.partial_g_u[-2])
        return grad_tuple

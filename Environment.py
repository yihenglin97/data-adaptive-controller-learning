# this is an abstract class of dynamical environment

class Environment:

    def __init__(self, sys_params, init_state):
        self.sys_params = sys_params
        self.state = init_state
        self.time_counter = 0

    # at the current time step, observe the current state and the context to decide the next action
    def observe(self):
        pass

    # at the current time step, evolve to the next state using the control action the user picks
    def step(self, control_action):
        pass

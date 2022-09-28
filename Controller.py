# this is an abstract class for parameterized controller

class Controller:

    def __init__(self, initial_param, buffer_length, learning_rate):
        self.param = initial_param
        self.learning_rate = learning_rate
        self.buffer_length = buffer_length
        self.buffer = []

    def decide_action(self, state, context):
        pass

    def update_param(self, grads):
        pass
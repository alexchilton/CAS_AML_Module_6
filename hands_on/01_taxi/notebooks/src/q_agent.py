import numpy as np
from pdb import set_trace as stop

class QAgent:
    def __init__(self, env, alpha, gamma):
        self.env = env
        self.q_table = np.zeros([env.observation_space.n, env.action_space.n])
        self.alpha = alpha
        self.gamma = gamma

    def get_action(self, state):
        # The original code incorrectly tried to convert the state to a tuple
        # This is unnecessary as the state in Taxi-v3 is already an integer
        # We should directly use the state as the index for the q_table
        # state = tuple(state)  # This line is incorrect and should be removed
        return np.argmax(self.q_table[state])  

    def update_parameters(self, state, action, reward, next_state):
        old_value = self.q_table[state, action]
        next_max = np.max(self.q_table[next_state])
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.q_table[state, action] = new_value

    def reset(self):
        """
        Sets q-values to zeros, which essentially means the agent does not know
        anything
        """
        self.q_table = np.zeros([self.env.observation_space.n, self.env.action_space.n])

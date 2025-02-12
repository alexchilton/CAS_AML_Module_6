import numpy as np
import random

from src.base_agent import BaseAgent

class SarsaAgent(BaseAgent):

    def __init__(self, env, alpha, gamma):

        self.env = env
        self.q_table = self._init_q_table()

        # hyper-parameters
        self.alpha = alpha
        self.gamma = gamma

    def _init_q_table(self) -> np.array:
        """
        Return numpy array with 3 dimensions.
        The first 2 dimensions are the state components, i.e. position, speed.
        The third dimension is the action.
        """
        # discretize state space from a continuous to discrete
        high = self.env.observation_space.high
        low = self.env.observation_space.low
        n_states = (high - low) * np.array([10, 100])
        n_states = np.round(n_states, 0).astype(int) + 1

        # table with q-values: n_states[0] * n_states[1] * n_actions
        return np.zeros([n_states[0], n_states[1], self.env.action_space.n])

    def _discretize_state(self, state):
        min_states = self.env.observation_space.low
        state_discrete = (state - min_states) * np.array([10, 100])
        return np.round(state_discrete, 0).astype(int)

    def get_action(self, state, epsilon=None):
        """"""
        if epsilon and random.uniform(0, 1) < epsilon:
            # Explore action space
            action = self.env.action_space.sample()
        else:
            # Exploit learned values
            state_discrete = self._discretize_state(state)
            action = np.argmax(self.q_table[state_discrete[0], state_discrete[1]])
        
        return action

    def state_to_index(self, state):
        """Convert a continuous state to a discrete index."""
        position_index = np.digitize(state[0], self.position_bins) - 1
        velocity_index = np.digitize(state[1], self.velocity_bins) - 1
        return position_index, velocity_index

    def update_parameters(self, state, action, reward, next_state, epsilon):
        """"""
        s = self._discretize_state(state)
        ns = self._discretize_state(next_state)
        na = self.get_action(next_state, epsilon)

        delta = self.alpha * (
                reward
                + self.gamma * self.q_table[ns[0], ns[1], na]
                - self.q_table[s[0], s[1], action]
        )
        self.q_table[s[0], s[1], action] += delta
        
    def act(self, state, epsilon=0.1):
        """
        Choose an action using epsilon-greedy policy.
        
        Args:
            state (tuple): The current state of the environment.
            epsilon (float): The probability of choosing a random action.
        
        Returns:
            int: The chosen action.
        """

        if np.random.rand() < epsilon:
            # Explore: choose a random action
            return self.env.action_space.sample()
        else:
            # Exploit: choose the action with the highest Q-value
            state_idx = self.state_to_index(state)
            action_idx = np.argmax(self.q_table[state_idx])
            return action_idx
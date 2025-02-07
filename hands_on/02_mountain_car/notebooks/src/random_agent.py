from src.base_agent import BaseAgent

class RandomAgent(BaseAgent):
    """
    This taxi driver selects actions randomly.
    You better not get into this taxi!
    """
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space

    def act(self, state, epsilon=0): # Add this method
        """Chooses an action randomly from the action space."""
        return self.action_space.sample()

    def get_action(self, state, epsilon) -> int:
        """
        No input arguments to this function.
        The agent does not consider the state of the environment when deciding
        what to do next.
        """
        return self.env.action_space.sample()

    def update_parameters(self, state, action, reward, next_state, epsilon):
        pass


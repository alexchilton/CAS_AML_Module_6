from typing import Tuple, List, Callable, Union, Optional
import random
from typing import Tuple
import numpy as np
import gym

from tqdm import tqdm

def train(agent, env, n_episodes=1000, epsilon=0.1):
    """
    Train the agent for a given number of episodes.

    Args:
        agent: The agent to train.
        env: The environment to train the agent in.
        n_episodes: The number of episodes to train the agent for.
        epsilon: The probability of taking a random action.

    Returns:
        A tuple containing the rewards and max positions for each episode.
    """

    rewards = []
    max_positions = []

    for episode in range(n_episodes):
        state = env.reset()[0]  # Reset the environment and get the initial state
        action = agent.get_action(state, epsilon)
        total_reward = 0
        max_position = -np.inf

        done = False
        while not done:
            next_state, reward, done, truncated, info = env.step(action)  # Update `env.step()` call
            next_action = agent.get_action(next_state, epsilon)
            agent.update(state, action, reward, next_state, next_action, done)  # Update `agent.update()` call

            state = next_state
            action = next_action
            total_reward += reward
            max_position = max(max_position, next_state[0])  # Update max position

        rewards.append(total_reward)
        max_positions.append(max_position)

        if (episode + 1) % 100 == 0:
            print(f"Episode: {episode + 1}, Reward: {total_reward}, Max Position: {max_position}")

    return rewards, max_positions

def evaluate(agent, env, n_episodes, epsilon=0.0):  # Add epsilon parameter with default value 0.0
    """
    Evaluate the agent for a given number of episodes.

    Args:
        agent: The agent to evaluate.
        env: The environment to evaluate the agent in.
        n_episodes: The number of episodes to evaluate the agent for.
        epsilon: The probability of taking a random action.

    Returns:
        A tuple containing the rewards and max positions for each episode.
    """
    rewards = []
    max_positions = []

    for episode in range(n_episodes):
        state = env.reset()[0]  # Reset the environment and get the initial state
        total_reward = 0
        max_position = -np.inf

        done = False
        while not done:
            action = agent.get_action(state, epsilon)  # Get action based on current policy
            next_state, reward, done, truncated, info = env.step(action)  # Take action and observe next state and reward
            state = next_state  # Update current state
            total_reward += reward
            max_position = max(max_position, next_state[0])  # Update max position

        rewards.append(total_reward)
        max_positions.append(max_position)

    return rewards, max_positions
    
if __name__ == '__main__':

    # environment
    import gym
    env = gym.make('MountainCar-v0')
    env._max_episode_steps = 1000

    # agent
    from src.sarsa_agent import SarsaAgent
    alpha = 0.1
    gamma = 0.6
    agent = SarsaAgent(env, alpha, gamma)

    rewards, max_positions = train(agent, env, n_episodes=100, epsilon=0.1)
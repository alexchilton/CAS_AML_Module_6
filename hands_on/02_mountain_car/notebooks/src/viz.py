from time import sleep
from argparse import ArgumentParser
from pdb import set_trace as stop

import pandas as pd
import gym

from src.config import SAVED_AGENTS_DIR

import numpy as np


def plot_policy(agent, positions: np.arange, velocities: np.arange, figsize = None):
    """"""
    data = []
    int2str = {
        0: 'Accelerate Left',
        1: 'Do nothing',
        2: 'Accelerate Right'
    }
    for position in positions:
        for velocity in velocities:

            state = np.array([position, velocity])
            action = int2str[agent.get_action(state)]

            data.append({
                'position': position,
                'velocity': velocity,
                'action': action,
            })

    data = pd.DataFrame(data)

    import seaborn as sns
    import matplotlib.pyplot as plt

    if figsize:
        plt.figure(figsize=figsize)

    colors = {
        'Accelerate Left': 'blue',
        'Do nothing': 'grey',
        'Accelerate Right': 'orange'
    }
    sns.scatterplot(x="position", y="velocity", hue="action", data=data,
                    palette=colors)

    plt.show()
    return data

def show_video(agent, env, sleep_sec=0.05, mode="rgb_array"): 
    """
    Shows a video of the agent acting in the environment.

    Args:
        agent: The agent to use.
        env: The environment to use.
        sleep_sec: The number of seconds to sleep between frames.
        mode: render mode of the environment
    """
    state = env.reset()
    # LAPADULA
    if mode == "human":
        state = state[0]
    # LAPADULA
    screen = env.render()
    done = False

    while not done:
        # Render the environment
        env.render()

        # Choose an action
        action = agent.get_action(state, epsilon=0)
        # LAPADULA
        # state, reward, done, info = env.step(action) # Original 
        state, reward, terminated, truncated, info = env.step(action) # Modified
        done = terminated or truncated
        if mode == "human":
            state = state[0]
        # LAPADULA
        # Sleep for a bit
        if sleep_sec:
            time.sleep(sleep_sec)


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--agent_file', type=str, required=True)
    parser.add_argument('--sleep_sec', type=float, required=False, default=0.1)
    args = parser.parse_args()

    from src.base_agent import BaseAgent
    agent_path = SAVED_AGENTS_DIR / args.agent_file
    agent = BaseAgent.load_from_disk(agent_path)

    env = gym.make('MountainCar-v0')
    env._max_episode_steps = 1000

    show_video(agent, env, sleep_sec=args.sleep_sec)









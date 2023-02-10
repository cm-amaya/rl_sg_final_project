import os
import pickle
import argparse
import numpy as np
import gymnasium as gym
from progress.bar import Bar
from datetime import datetime
from lunar_lander.agents.core import BaseAgent
from lunar_lander.agents.q_learning_agent import QLearningAgent
from lunar_lander.environment import Enviroment

parser = argparse.ArgumentParser()
parser.add_argument('--episodes', type=int, default=200)
parser.add_argument('--agent', type=str, default="QLearning")
parser.add_argument('--render', type=bool, default=False)


def save_results(agent_type,
                 q_values,
                 steps,
                 n_rewards,
                 n_state_visits,
                 n_episodes):
    result_folder = 'saves/results'
    if not os.path.exists(result_folder):
        os.mkdir(result_folder)
    now = datetime.now().strftime("%m%d%Y_%H%M%S")
    results_path = os.path.join(result_folder,
                                f'{agent_type}_{n_episodes}_{now}.pickle')
    results = {'agent_type': agent_type,
               'q_values': q_values,
               'steps': steps,
               'n_rewards': n_rewards,
               'n_state_visits': n_state_visits,
               'episodes': n_episodes}
    with open(results_path, 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


def episode(agent: BaseAgent, env: gym.Env, render: bool = False):
    observation, info = env.reset()
    action = agent.agent_start(observation)
    if render:
        env.render()
    terminated = False
    steps = 0
    rewards = 0.0
    state_visits = np.zeros(48)
    while not terminated:
        steps += 1
        observation, reward, terminated, truncated, info = env.step(action)
        rewards += reward
        state_visits[observation] += 1
        if terminated or truncated:
            agent.agent_end(reward)
            observation, info = env.reset()
            break
        action = agent.agent_step(observation, reward)
    return steps, rewards, state_visits


def main():
    args = parser.parse_args()
    agent_type = args.agent
    episodes = args.episodes
    render = args.render
    saves_folder = 'saves'
    if not os.path.exists(saves_folder):
        os.mkdir(saves_folder)
    map = Enviroment()
    if agent_type == "QLearning":
        agent = QLearningAgent()
    agent_info = {"num_actions": 4,
                  "num_states": 16,
                  "epsilon": 0.1,
                  "step_size": 0.1,
                  "discount": 1.0}
    agent.agent_init(agent_info)
    n_steps = []
    n_rewards = []
    n_state_visits = []
    bar = Bar('Processing', max=20)
    for _ in range(episodes):
        steps, rewards, state_visits = episode(agent, map.env, render)
        n_steps.append(steps)
        n_rewards.append(rewards)
        n_state_visits.append(state_visits)
        bar.next()
    bar.finish()
    agent.agent_save()
    save_results(agent_type,
                 agent.q_values,
                 n_steps,
                 n_rewards,
                 n_state_visits,
                 episodes)
    map.env.close()


if __name__ == "__main__":
    main()

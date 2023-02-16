import os
import shutil
import argparse
import numpy as np
from progress.bar import Bar
from lunar_lander.rl_glue import RLGlue
from lunar_lander.agents.core import BaseAgent
from lunar_lander.environments.core import BaseEnvironment
from lunar_lander.agents.action_value_agent import ActionValueAgent
from lunar_lander.environments.lunar_lander_env import LunarLanderEnvironment
from lunar_lander.environments.lunar_lander_alt_env import (
    LunarLanderAltEnvironment
)
parser = argparse.ArgumentParser()
parser.add_argument("--episodes", type=int, default=200)
parser.add_argument("--mode", type=str, required=False, default="Normal")
parser.add_argument("--runs", type=int, default=1)


def run_experiment(
    environment: BaseEnvironment,
    agent: BaseAgent,
    agent_parameters,
    experiment_parameters,
    mode
):
    rl_glue = RLGlue(environment, agent)
    # save sum of reward at the end of each episode
    agent_sum_reward = np.zeros(
        (experiment_parameters["num_runs"],
         experiment_parameters["num_episodes"])
    )
    env_info = {}
    agent_info = agent_parameters
    # one agent setting
    bar = Bar("Processing Run", max=experiment_parameters["num_runs"])
    for run in range(1, experiment_parameters["num_runs"] + 1):
        agent_info["seed"] = run
        agent_info["network_config"]["seed"] = run
        env_info["seed"] = run
        rl_glue.rl_init(agent_info, env_info)
        for episode in range(1, experiment_parameters["num_episodes"] + 1):
            # run episode
            rl_glue.rl_episode(experiment_parameters["timeout"])
            episode_reward = rl_glue.rl_agent_message("get_sum_reward")
            agent_sum_reward[run - 1, episode - 1] = episode_reward
        bar.next()
    bar.finish()
    save_name = "{}".format(rl_glue.agent.name)
    if not os.path.exists("results"):
        os.makedirs("results")
    np.save("results/sum_reward_{}_{}".format(save_name, mode),
            agent_sum_reward)
    shutil.make_archive("results", "zip", "results")
    rl_glue.rl_cleanup()


def main():
    args = parser.parse_args()
    mode = args.mode
    episodes = args.episodes
    runs = args.runs
    saves_folder = "saves"
    if not os.path.exists(saves_folder):
        os.mkdir(saves_folder)
    if mode == "Normal":
        current_env = LunarLanderEnvironment
    else:
        current_env = LunarLanderAltEnvironment
    current_agent = ActionValueAgent
    experiment_parameters = {
        "num_runs": runs,
        "num_episodes": episodes,
        "timeout": 1000,
    }
    agent_parameters = {
        "network_config": {
            "state_dim": 8,
            "num_hidden_units": 256,
            "num_actions": 4
            },
        "optimizer_config": {
            "step_size": 1e-3,
            "beta_m": 0.9,
            "beta_v": 0.999,
            "epsilon": 1e-8,
            },
        "replay_buffer_size": 50000,
        "minibatch_sz": 8,
        "num_replay_updates_per_step": 4,
        "gamma": 0.99,
        "tau": 0.001,
    }
    run_experiment(current_env,
                   current_agent,
                   agent_parameters,
                   experiment_parameters,
                   mode)


if __name__ == "__main__":
    main()

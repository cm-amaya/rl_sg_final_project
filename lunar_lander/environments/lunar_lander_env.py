import os
import pickle
import gymnasium as gym
from lunar_lander.environments.core import BaseEnvironment


class LunarLanderEnvironment(BaseEnvironment):
    close = None
    def __init__(self, env_info={}):
        self.env = gym.make("LunarLander-v2")
        self.env_info = env_info

    def env_start(self):
        """
        The first method called when the experiment starts, called before the
        agent starts.
        Returns:
            The first state observation from the environment.
        """
        reward = 0.0
        observation, _ = self.env.reset()
        is_terminal = False
        self.reward_obs_term = (reward, observation, is_terminal)
        return self.reward_obs_term[1]

    def env_step(self, action):
        """A step taken by the environment.
        Args:
            action: The action taken by the agent
        Returns:
            (float, state, Boolean): a tuple of the reward, state observation,
                and boolean indicating if it's terminal.
        """
        last_state = self.reward_obs_term[1]
        current_state, reward, is_terminal, truncated, info = self.env.step(action)
        self.reward_obs_term = (reward, current_state, is_terminal)
        return self.reward_obs_term

    def env_cleanup(self):
        self.env.close()

    def env_save(self):
        with open(self._map_path, 'wb') as handle:
            pickle.dump(self._env, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def close(self):
        self.env.close()

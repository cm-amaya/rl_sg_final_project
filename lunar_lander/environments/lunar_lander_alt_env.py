import pickle
import numpy as np
import gymnasium as gym
from lunar_lander.environments.core import BaseEnvironment

VIEWPORT_W = 600
VIEWPORT_H = 400
FPS = 50
SCALE = 30.0
LEG_DOWN = 18
W = VIEWPORT_W / SCALE
H = VIEWPORT_H / SCALE
HELIPAD_Y = H / 4


def get_position(state):
    pos_x = state[0]*(W/2) + (W/2)
    pos_y = state[1]*(H/2) + (HELIPAD_Y+LEG_DOWN/SCALE)
    return pos_x, pos_y

def get_speed(state):
    vel_x = state[2]*FPS/(W/2)
    vel_y = state[3]*FPS/(H/2)
    return vel_x, vel_y

def reward_function_alternative(state):
    reward = 100.0
    pos_x, pos_y = get_position(state)
    vel_x, vel_y = get_position(state)
    # angle should point towards center
    # more than 0.4 radians (22 degrees) is bad
    angle_targ = state[0] * 0.5 + state[2] * 1.0
    if angle_targ > 0.4 or angle_targ < -0.4:
        reward -= 10.0

    if abs(state[0]) >= 1.0:
        reward = -100
    if state[6] == 1.0 and state[7] == 1.0:
        reward = 100
    return reward


class LunarLanderAltEnvironment(BaseEnvironment):
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

    def close(self):
        self.env_cleanup()

    def env_save(self):
        with open(self._map_path, 'wb') as handle:
            pickle.dump(self._env, handle, protocol=pickle.HIGHEST_PROTOCOL)

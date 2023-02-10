import os
import pickle
import numpy as np
from datetime import datetime
from lunar_lander.agents.core import BaseAgent


class QLearningAgent(BaseAgent):

    def __init__(self):
        pass

    def agent_init(self, agent_info: dict):
        self.num_states = agent_info["num_states"]
        self.num_actions = agent_info["num_actions"]
        self.gamma = agent_info.get("discount", 0.95)
        self.step_size = agent_info.get("step_size", 0.1)
        self.epsilon = agent_info.get("epsilon", 0.1)
        self._q_values = np.zeros((self.num_states, self.num_actions))
        self.actions = np.arange(self.num_actions)
        self.past_action = -1
        self.past_state = -1

    @property
    def q_values(self):
        return self._q_values

    def agent_start(self, state):
        """The first method called when the experiment starts,
        called after the environment starts.
        Args:
            state (Numpy array): the state from the
                environment's env_start function.
        Returns:
            (int) the first action the agent takes.
        """
        action = self.choose_action_egreedy(state)
        self.past_state = state
        self.past_action = action
        return self.past_action

    def agent_step(self, state, reward):
        prev_Q = self.q_values[self.past_state][self.past_action]
        self._q_values[self.past_state][self.past_action] += self.step_size \
            * (reward + self.gamma * np.max(self.q_values[state])-prev_Q)
        action = self.choose_action_egreedy(state)
        self.past_state = state
        self.past_action = action
        return self.past_action

    def agent_end(self, reward):
        prev_Q = self.q_values[self.past_state][self.past_action]
        self._q_values[self.past_state][self.past_action] += self.step_size \
            * (reward - prev_Q)

    def argmax(self, q_values):
        """argmax with random tie-breaking
        Args:
            q_values (Numpy array): the array of action values
        Returns:
            action (int): an action with the highest value
        """
        top = float("-inf")
        ties = []

        for i in range(len(q_values)):
            if q_values[i] > top:
                top = q_values[i]
                ties = []

            if q_values[i] == top:
                ties.append(i)

        return np.random.choice(ties)

    def choose_action_egreedy(self, state):
        """returns an action using an epsilon-greedy policy.
        Args:
            state (int): coordinates of the agent
        Returns:
            The action taken w.r.t. the aforementioned epsilon-greedy policy
        """

        if np.random.random() < self.epsilon:
            action = np.random.choice(self.actions)
        else:
            values = self._q_values[state]
            action = self.argmax(values)
        return action

    def agent_save(self):
        agents_folder = 'saves/agent_values'
        if not os.path.exists(agents_folder):
            os.mkdir(agents_folder)
        now = datetime.now().strftime("%m%d%Y_%H%M%S")
        results_path = os.path.join(agents_folder,
                                    f'QLearningAgent_{now}.pickle')
        results = {'q_values': self._q_values}
        with open(results_path, 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

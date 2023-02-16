import numpy as np
from copy import deepcopy


class Adam():
    def __init__(self,
                 layer_sizes,
                 optimizer_info):
        self.layer_sizes = layer_sizes
        # Specify Adam algorithm's hyper parameters
        self.step_size = optimizer_info.get("step_size")
        self.beta_m = optimizer_info.get("beta_m")
        self.beta_v = optimizer_info.get("beta_v")
        self.epsilon = optimizer_info.get("epsilon")
        # Initialize Adam algorithm's m and v
        self.m = [dict() for i in range(1, len(self.layer_sizes))]
        self.v = [dict() for i in range(1, len(self.layer_sizes))]
        for i in range(0, len(self.layer_sizes) - 1):
            self.m[i]["W"] = np.zeros((self.layer_sizes[i],
                                       self.layer_sizes[i+1]))
            self.m[i]["b"] = np.zeros((1, self.layer_sizes[i+1]))
            self.v[i]["W"] = np.zeros((self.layer_sizes[i],
                                       self.layer_sizes[i+1]))
            self.v[i]["b"] = np.zeros((1, self.layer_sizes[i+1]))
        self.beta_m_product = self.beta_m
        self.beta_v_product = self.beta_v

    def update_weights(self, weights, td_errors_times_gradients):
        """
        Args:
            weights (Array of dictionaries): The weights of the neural network.
            td_errors_times_gradients (Array of dictionaries): The gradient of
            the action-values with respect to the network's weights times the
            TD-error
        Returns:
            The updated weights (Array of dictionaries).
        """
        for i in range(len(weights)):
            for param in weights[i].keys():
                # update self.m and self.v
                self.m[i][param] = self.beta_m * self.m[i][param]
                + (1 - self.beta_m) * td_errors_times_gradients[i][param]
                self.v[i][param] = self.beta_v * self.v[i][param] + \
                    (1 - self.beta_v) * (td_errors_times_gradients[i][param]
                                         ** 2)
                # compute m_hat and v_hat
                m_hat = self.m[i][param] / (1 - self.beta_m_product)
                v_hat = self.v[i][param] / (1 - self.beta_v_product)
                # update weights
                weight_update = self.step_size * m_hat / (np.sqrt(v_hat) +
                                                          self.epsilon)
                weights[i][param] = weights[i][param] + weight_update
        self.beta_m_product *= self.beta_m
        self.beta_v_product *= self.beta_v
        return weights


class ActionValueNetwork:
    def __init__(self, network_config):
        self.state_dim = network_config.get("state_dim")
        self.num_hidden_units = network_config.get("num_hidden_units")
        self.num_actions = network_config.get("num_actions")
        self.rand_generator = np.random.RandomState(network_config.get("seed"))
        self.layer_sizes = np.array([self.state_dim,
                                     self.num_hidden_units,
                                     self.num_actions])
        self.weights = [dict() for i in range(0, len(self.layer_sizes) - 1)]
        for i in range(0, len(self.layer_sizes) - 1):
            self.weights[i]['W'] = self.init_saxe(self.layer_sizes[i],
                                                  self.layer_sizes[i + 1])
            self.weights[i]['b'] = np.zeros((1, self.layer_sizes[i + 1]))

    def get_action_values(self, s):
        """
        Args:
            s (Numpy array): The state.
        Returns:
            The action-values (Numpy array) calculated using the network's
                weights.
        """
        W0, b0 = self.weights[0]['W'], self.weights[0]['b']
        psi = np.dot(s, W0) + b0
        x = np.maximum(psi, 0)
        W1, b1 = self.weights[1]['W'], self.weights[1]['b']
        q_vals = np.dot(x, W1) + b1

        return q_vals

    def get_TD_update(self, s, delta_mat):
        """
        Args:
            s (Numpy array): The state.
            delta_mat (Numpy array): A 2D array of shape
                (batch_size, num_actions). Each row of delta_mat correspond to
                one state in the batch. Each row has only one non-zero element
                which is the TD-error corresponding to the action taken.
        Returns:
            The TD update (Array of dictionaries with gradient times TD errors)
            for the network's weights
        """
        W0, b0 = self.weights[0]['W'], self.weights[0]['b']
        W1, _ = self.weights[1]['W'], self.weights[1]['b']
        psi = np.dot(s, W0) + b0
        x = np.maximum(psi, 0)
        dx = (psi > 0).astype(float)
        td_update = [dict() for i in range(len(self.weights))]
        v = delta_mat
        td_update[1]['W'] = np.dot(x.T, v) * 1. / s.shape[0]
        td_update[1]['b'] = np.sum(v, axis=0, keepdims=True) * 1. / s.shape[0]
        v = np.dot(v, W1.T) * dx
        td_update[0]['W'] = np.dot(s.T, v) * 1. / s.shape[0]
        td_update[0]['b'] = np.sum(v, axis=0, keepdims=True) * 1. / s.shape[0]
        return td_update

    def init_saxe(self, rows, cols):
        """
        Args:
            rows (int): number of input units for layer.
            cols (int): number of output units for layer.
        Returns:
            NumPy Array consisting of weights for the layer based on the
                initialization in Saxe et al.
        """
        tensor = self.rand_generator.normal(0, 1, (rows, cols))
        if rows < cols:
            tensor = tensor.T
        tensor, r = np.linalg.qr(tensor)
        d = np.diag(r, 0)
        ph = np.sign(d)
        tensor *= ph

        if rows < cols:
            tensor = tensor.T
        return tensor

    def get_weights(self):
        """
        Returns:
            A copy of the current weights of this network.
        """
        return deepcopy(self.weights)

    def set_weights(self, weights):
        """
        Args:
            weights (list of dictionaries): Consists of weights that this
                network will set as its own weights.
        """
        self.weights = deepcopy(weights)


class ReplayBuffer:
    def __init__(self, size, minibatch_size, seed):
        """
        Args:
            size (integer): The size of the replay buffer.
            minibatch_size (integer): The sample size.
            seed (integer): The seed for the random number generator.
        """
        self.buffer = []
        self.minibatch_size = minibatch_size
        self.rand_generator = np.random.RandomState(seed)
        self.max_size = size

    def append(self, state, action, reward, terminal, next_state):
        """
        Args:
            state (Numpy array): The state.
            action (integer): The action.
            reward (float): The reward.
            terminal (integer): 1 if the next state is a terminal state and
                otherwise.
            next_state (Numpy array): The next state.
        """
        if len(self.buffer) == self.max_size:
            del self.buffer[0]
        self.buffer.append([state, action, reward, terminal, next_state])

    def sample(self):
        """
        Returns:
            A list of transition tuples including state, action, reward,
            terminal, and next_state
        """
        idxs = self.rand_generator.choice(np.arange(len(self.buffer)),
                                          size=self.minibatch_size)
        return [self.buffer[idx] for idx in idxs]

    def size(self):
        return len(self.buffer)


def softmax(action_values, tau=1.0):
    """
    Args:
        action_values (Numpy array): A 2D array of shape
            (batch_size, num_actions). The action-values computed by an
            action-value network.
        tau (float): The temperature parameter scalar.
    Returns:
        A 2D array of shape (batch_size, num_actions). Where each column is a
        probability distribution over the actions representing the policy.
    """
    preferences = action_values / tau
    max_preference = np.max(preferences, axis=1)
    reshaped_max_preference = max_preference.reshape((-1, 1))
    exp_preferences = np.exp(preferences - reshaped_max_preference)
    sum_of_exp_preferences = np.sum(exp_preferences, axis=1)
    reshaped_sum_of_exp_preferences = sum_of_exp_preferences.reshape((-1, 1))
    action_probs = exp_preferences / reshaped_sum_of_exp_preferences
    action_probs = action_probs.squeeze()
    return action_probs


def get_td_error(states,
                 next_states,
                 actions,
                 rewards,
                 discount,
                 terminals,
                 network,
                 current_q,
                 tau):
    """
    Args:
        states (Numpy array): The batch of states with the shape
            (batch_size, state_dim).
        next_states (Numpy array): The batch of next states with the shape
            (batch_size, state_dim).
        actions (Numpy array): The batch of actions with the shape
            (batch_size,).
        rewards (Numpy array): The batch of rewards with the shape
            (batch_size,).
        discount (float): The discount factor.
        terminals (Numpy array): The batch of terminals with the shape
            (batch_size,).
        network (ActionValueNetwork): The latest state of the network that is
            getting replay updates.
        current_q (ActionValueNetwork): The fixed network used for computing
        the targets, and particularly, the action-values at the next-states.
    Returns:
        The TD errors (Numpy array) for actions taken, of shape (batch_size,)
    """
    q_next_mat = current_q.get_action_values(next_states)
    probs_mat = softmax(q_next_mat, tau)
    v_next_vec = np.sum(probs_mat * q_next_mat, axis=1) * (1 - terminals)
    target_vec = rewards + discount * v_next_vec
    q_mat = network.get_action_values(states)
    batch_indices = np.arange(q_mat.shape[0])
    q_vec = q_mat[batch_indices, actions]
    delta_vec = target_vec - q_vec
    return delta_vec


def optimize_network(experiences,
                     discount,
                     optimizer,
                     network,
                     current_q,
                     tau):
    """
    Args:
        experiences (Numpy array): The batch of experiences including the
            states, actions, rewards, terminals, and next_states.
        discount (float): The discount factor.
        network (ActionValueNetwork): The latest state of the network that is
            getting replay updates.
        current_q (ActionValueNetwork): The fixed network used for computing
        the targets, and particularly, the action-values at the next-states.
    """

    states, actions, rewards, terminals, next_states = map(list,
                                                           zip(*experiences))
    states = np.concatenate(states)
    next_states = np.concatenate(next_states)
    rewards = np.array(rewards)
    terminals = np.array(terminals)
    batch_size = states.shape[0]

    delta_vec = get_td_error(states,
                             next_states,
                             actions,
                             rewards,
                             discount,
                             terminals,
                             network,
                             current_q,
                             tau)
    batch_indices = np.arange(batch_size)
    delta_mat = np.zeros((batch_size, network.num_actions))
    delta_mat[batch_indices, actions] = delta_vec
    td_update = network.get_TD_update(states, delta_mat)
    weights = optimizer.update_weights(network.get_weights(), td_update)
    network.set_weights(weights)

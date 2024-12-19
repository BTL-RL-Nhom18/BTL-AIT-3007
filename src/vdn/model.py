import numpy as np
import collections
import torch
import torch.nn as nn
from utils import compute_output_dim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    def __init__(self, buffer_limit):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        """Update buffer with a new transition
        :param transition: tuple of (state, action, reward, next_state, done)
        """
        self.buffer.append(transition)

    def sample_chunk(self, batch_size, chunk_size):
        """Sample a batch of chunk_size transitions from the buffer
        :param batch_size: number of transitions to sample
        :param chunk_size: length of horizon of each batch
        :return: tuple of (states, actions, rewards, next_states, dones),
        their shapes are respectively:
        [batch_size, chunk_size, n_agents, ...obs_shape],
        [batch_size, chunk_size, n_agents],
        [batch_size, chunk_size, n_agents],
        [batch_size, chunk_size, n_agents, ...obs_shape],
        [batch_size, chunk_size, n_agents]
        """
        start_idx = np.random.randint(0, len(self.buffer) - chunk_size, batch_size)
        s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], []

        for idx in start_idx:
            for chunk_step in range(idx, idx + chunk_size):
                # (state, action, reward, next_state, done) * num_agent
                s, a, r, s_prime, done = self.buffer[chunk_step]
                s_lst.append(s)
                a_lst.append(a)
                r_lst.append(r)
                s_prime_lst.append(s_prime)
                done_lst.append(done)
        num_agents = len(s_lst[0])
        obs_shape = s_lst[0][0].shape

        s_lst = np.array(s_lst).reshape(batch_size, chunk_size, num_agents, *obs_shape)
        a_lst = np.array(a_lst).reshape(batch_size, chunk_size, num_agents)
        r_lst = np.array(r_lst).reshape(batch_size, chunk_size, num_agents)
        s_prime_lst = np.array(s_prime_lst).reshape(batch_size, chunk_size, num_agents, *obs_shape)
        done_lst = np.array(done_lst, dtype=bool).reshape(batch_size, chunk_size, num_agents)
        return (
            torch.tensor(s_lst, dtype=torch.float32).to(device),
            torch.tensor(a_lst, dtype=torch.float32).to(device),
            torch.tensor(r_lst, dtype=torch.float32).to(device),
            torch.tensor(s_prime_lst, dtype=torch.float32).to(device),
            torch.tensor(done_lst, dtype=torch.float32).to(device)
        )

    def size(self):
        return len(self.buffer)
    
class VdnQNet(nn.Module):
    def __init__(self, agents, observation_spaces, action_spaces, recurrent=False):
        super(VdnQNet, self).__init__()
        self.agents = agents
        self.num_agents = len(agents)
        self.recurrent = recurrent
        self.hx_size = 32   # latent repr size
        self.n_obs = observation_spaces[agents[0]].shape    # observation space size of agents
        self.n_act = action_spaces[agents[0]].n  # action space size of agents

        stride1, stride2 = 1, 1
        padding1, padding2 = 1, 1
        kernel_size1, kernel_size2 = 3, 3
        pool_kernel_size, pool_stride = 2, 2

        height = self.n_obs[0]  # n_obs is a tuple (height, width, channels)
        out_dim1 = compute_output_dim(height, kernel_size1, stride1, padding1) // pool_stride
        out_dim2 = compute_output_dim(out_dim1, kernel_size2, stride2, padding2) // pool_stride

        # Compute the final flattened size
        flattened_size = out_dim2 * out_dim2 * 64
        self.feature_cnn = nn.Sequential(
            nn.Conv2d(self.n_obs[2], 32, kernel_size=kernel_size1, stride=stride1, padding=padding1),
            nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride),
            nn.Conv2d(32, 64, kernel_size=kernel_size2, stride=stride2, padding=padding2),
            nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride),
            nn.Flatten(),
            nn.Linear(flattened_size, self.hx_size),
            nn.ReLU()
        )
        if recurrent:
            self.gru =  nn.GRUCell(self.hx_size, self.hx_size)  # shape: hx_size, hx_size
        self.q_val = nn.Linear(self.hx_size, self.n_act)    # shape: hx_size, n_actions

    def forward(self, obs, hidden):
        """Predict q values for each agent's actions in the batch
        :param obs: [batch_size, num_agents, ...n_obs]
        :param hidden: [batch_size, num_agents, hx_size]
        :return: q_values: [batch_size, num_agents, n_actions], hidden: [batch_size, num_agents, hx_size]
        """
        obs = obs.to(device)
        hidden = hidden.to(device)
        
        batch_size, num_agents, height, width, channels = obs.shape
        obs = obs.permute(0, 1, 4, 2, 3)  # (batch_size, num_agents, channels, height, width)
        obs = obs.reshape(batch_size * num_agents, channels, height, width)  # (batch_size * num_agents, channels, height, width)
        
        x = self.feature_cnn(obs)  # (batch_size * num_agents, hx_size)
        
        if self.recurrent:
            hidden = hidden.reshape(batch_size * num_agents, -1)  # (batch_size * num_agents, hx_size)
            x = self.gru(x, hidden)  # (batch_size * num_agents, hx_size)
        
        q_values = self.q_val(x)  # (batch_size * num_agents, n_actions)
        
        q_values = q_values.view(batch_size, num_agents, -1)  # (batch_size, num_agents, n_actions)
        
        if self.recurrent:
            next_hidden = x.view(batch_size, num_agents, -1)  # (batch_size, num_agents, hx_size)
        else:
            next_hidden = hidden.view(batch_size, num_agents, -1)
        
        return q_values, next_hidden

    def sample_action(self, obs, hidden, epsilon=1e3):
        """Choose action with epsilon-greedy policy, for each agent in the batch
        :param obs: a batch of observations, [batch_size, num_agents, n_obs]
        :param hidden: a batch of hidden states, [batch_size, num_agents, hx_size]
        :param epsilon: exploration rate
        :return: actions: [batch_size, num_agents], hidden: [batch_size, num_agents, hx_size]
        """
        obs = obs.to(device)
        hidden = hidden.to(device)
        
        q_values, hidden = self.forward(obs, hidden)    # [batch_size, num_agents, n_actions], [batch_size, num_agents, hx_size]
        # epsilon-greedy action selection: choose random action with epsilon probability
        mask = (torch.rand((q_values.shape[0],), device=device) <= epsilon)  # [batch_size]
        actions = torch.empty((q_values.shape[0], q_values.shape[1]), device=device)  # [batch_size, num_agents]
        actions[mask] = torch.randint(0, q_values.shape[2], actions[mask].shape, device=device).float()
        actions[~mask] = q_values[~mask].argmax(dim=2).float()  # choose action with max q value
        return actions, hidden   # [batch_size, num_agents], [batch_size, num_agents, hx_size]

    def init_hidden(self, batch_size=1):
        return torch.zeros((batch_size, self.num_agents, self.hx_size), device=device)
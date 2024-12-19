import numpy as np
import supersuit
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import random
import os

from rewards import _calc_reward
from src.rnn_agent.rnn_agent import ReplayBufferGRU

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class ReplayBuffer(ReplayBufferGRU):
    """ 
    Replay buffer for agent with GRU network additionally storing previous action, 
    initial input hidden state and output hidden state of GRU.
    And each sample contains the whole episode instead of a single step.
    'hidden_in' and 'hidden_out' are only the initial hidden state for each episode, for GRU initialization.

    """
    def push(self, hidden_in, hidden_out, observation, state, next_state, action, reward, next_observation):      
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (
            hidden_in, hidden_out, observation, state, next_state, action, reward, next_observation)
        self.position = int((self.position + 1) %
                            self.capacity)  # as a ring buffer

    def sample(self, batch_size):
        obs_lst, a_lst, r_lst, nobs_lst, hi_lst, ho_lst, s_lst, ns_lst = [], [], [], [], [], [], [], []
        batch = random.sample(self.buffer, batch_size)
        min_seq_len = float('inf')
        for sample in batch:
            h_in, h_out, observation, state, next_state, action, reward, next_observation = sample
            min_seq_len = min(len(observation), min_seq_len)
            hi_lst.append(h_in) # h_in: (1, batch_size=1, n_agents, hidden_size)
            ho_lst.append(h_out)
        hi_lst = torch.cat(hi_lst, dim=-3).detach()  # cat along the batch dim
        ho_lst = torch.cat(ho_lst, dim=-3).detach()

        # strip sequence length
        for sample in batch:
            h_in, h_out, observation, state, next_state, action, reward, next_observation = sample
            sample_len = len(observation)
            start_idx = int((sample_len - min_seq_len)/2)
            end_idx = start_idx+min_seq_len
            obs_lst.append(observation[start_idx:end_idx])
            s_lst.append(state[start_idx:end_idx])
            ns_lst.append(next_state[start_idx:end_idx])
            a_lst.append(action[start_idx:end_idx])
            r_lst.append(reward[start_idx:end_idx])
            nobs_lst.append(next_observation[start_idx:end_idx])
        return hi_lst, ho_lst, obs_lst, s_lst, ns_lst, a_lst, r_lst, nobs_lst

class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super(CNNFeatureExtractor, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(5, 5, 3),
            nn.BatchNorm2d(5),
            nn.ReLU(),
            nn.Conv2d(5, 5, 3),
            nn.BatchNorm2d(5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        
    def get_output_dim(self, size):
        # Tính kích thước output sau khi qua CNN
        # Conv2d(kernel=3): h,w -> h-2,w-2
        # MaxPool2d(2): h,w -> h//2,w//2
        h, w = size
        h = (h - 2) - 2  # Sau 2 lớp Conv2d
        w = (w - 2) - 2
        h = h // 2  # Sau MaxPool2d
        w = w // 2
        return h * w * 5  # 5 là số channels output

    def forward(self, x):
        # x shape: [..., H, W, 5]
        orig_shape = x.shape[:-3]
        h, w = x.shape[-3:-1]
        
        x = x.view(-1, h, w, 5)
        x = x.permute(0, 3, 1, 2)  # [..., 5, H, W]
        x = self.cnn(x)
        x = x.reshape(*orig_shape, -1)  # Flatten CNN output
        return x
    
class RNNAgent(nn.Module):
    '''
    @brief:
        evaluate Q value given a state and the action
    '''

    def __init__(self, obs_dim, action_shape, num_actions, hidden_size, epsilon):
        super(RNNAgent, self).__init__()

        self.obs_dim = obs_dim
        self.action_shape = action_shape
        self.num_actions = num_actions
        self.epsilon = epsilon
        
        self.feature_extractor = CNNFeatureExtractor()

        self.linear1 = nn.Linear(obs_dim, hidden_size) #80+21 -> 64
        self.linear2 = nn.Linear(hidden_size, hidden_size)  #64 -> 64
        self.rnn = nn.GRU(hidden_size, hidden_size) #64 -> 64
        self.linear3 = nn.Linear(hidden_size, hidden_size) #64 -> 64
        self.linear4 = nn.Linear(hidden_size, action_shape*num_actions) #64 -> 21

    def forward(self, state, hidden_in):
        '''
        @params:
            state: [#batch, #sequence, #agent, #n_feature]
            action: [#batch, #sequence, #agent, action_shape]
        @return:
            qs: [#batch, #sequence, #agent, action_shape, num_actions]
        '''
        # Check the shape of state to determine if it's training or inference
        if len(state.shape) == 6:
            # Training mode
            bs, seq_len, n_agents, _, _, _ = state.shape
        elif len(state.shape) == 5:
            # Inference mode
            bs, seq_len, _, _, _ = state.shape
            n_agents = 1  # Set n_agents to 1 for inference
            state = state.unsqueeze(2)  # Add agent dimension
        else:
            raise ValueError("Invalid state shape. Expected 5 or 6 dimensions.")
                
        # Feature extraction from 2D observation
        state = self.feature_extractor(state)  # [batch, sequence, agents, cnn_features]
        state = state.permute(1, 0, 2, 3)

        # Reshape for RNN input
        x = state.reshape(seq_len, bs*n_agents, -1)  # [sequence, batch*agents, features]   
        hidden_in = hidden_in.view(1, bs*n_agents, -1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x,  hidden = self.rnn(x, hidden_in)
        x = F.relu(self.linear3(x))
        x = self.linear4(x) # [#sequence, #batch, #agents, #action_shape*#actions]
        # [#sequence, #batch, #agent, #head * #action]
        x = x.view(seq_len, bs, n_agents, self.action_shape, self.num_actions)
        hidden = hidden.view(1, bs, n_agents, -1)
        # categorical over the discretized actions
        qs = F.softmax(x, dim=-1)
        qs = qs.permute(1, 0, 2, 3, 4)  # permute back [#batch, #sequence, #agents, #action_shape, #actions]

        return qs, hidden
    
    def get_action(self, state, hidden_in):
        '''
        @brief:
            for each distributed agent, generate action for one step given input data
        @params:
            state: [n_agents, n_feature]
        '''
        state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(device) # add #sequence and #batch: [[#batch, #sequence, n_agents, n_feature]]
        hidden_in = hidden_in.unsqueeze(1) # add #batch: [#batch, n_agents, hidden_dim]
        agent_outs, hidden_out = self.forward(state, hidden_in)  # agents_out: [#batch, #sequence, n_agents, action_shape, action_dim]; hidden_out same as hidden_in
        agent_outs = agent_outs.squeeze(0).squeeze(0)  # Remove batch and sequence dims -> [n_agents, action_shape, action_dim]
        action = np.zeros((agent_outs.shape[0], agent_outs.shape[1]), dtype=np.int64)
        
        # Process each agent independently
        for agent_idx in range(agent_outs.shape[0]):
            for action_idx in range(agent_outs.shape[1]):
                if np.random.rand() < self.epsilon:
                    # Random action for this agent
                    dist = Categorical(agent_outs[agent_idx, action_idx])
                    action[agent_idx, action_idx] = dist.sample().cpu().numpy()
                else:
                    # Greedy action for this agent
                    action[agent_idx, action_idx] = torch.argmax(agent_outs[agent_idx, action_idx]).cpu().numpy()

        # Clear unnecessary tensors
        del state, hidden_in, agent_outs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
                
        return action, hidden_out

class QMix(nn.Module):
    def __init__(self, state_dim, n_agents, action_shape, embed_dim=64, hypernet_embed=128, abs=True):
        """
        Critic network class for Qmix. Outputs centralized value function predictions given independent q value.
        :param args: (argparse) arguments containing relevant model information.
        """
        super(QMix, self).__init__()

        self.n_agents = n_agents
        self.state_dim = state_dim # features
        self.action_shape = action_shape

        self.embed_dim = embed_dim
        self.hypernet_embed = hypernet_embed
        self.abs = abs
        
        self.feature_extractor = CNNFeatureExtractor()

        self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim, self.hypernet_embed),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(self.hypernet_embed, self.action_shape * self.embed_dim * self.n_agents))
        self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, self.hypernet_embed),
                                            nn.ReLU(inplace=True),
                                           nn.Linear(self.hypernet_embed, self.embed_dim))

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(
            self.state_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(inplace=True),
                               nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, states):
        """
        Compute actions from the given inputs.
        @params:
            agent_qs: [#batch, #sequence, #agent, #action_shape]
            states: [#batch, #sequence, #agent, #features*action_shape] [2, 1000, (45, 45, 5)]
        :param agent_qs: q value inputs into network [batch_size, #agent, action_shape]
        :param states: state observation.
        :return q_tot: (torch.Tensor) return q-total .
        """
        bs = agent_qs.size(0)
        
        # Feature extraction from 2D observation
        states = self.feature_extractor(states)
    
        states = states.reshape(-1, self.state_dim)  # [#batch*#sequence, action_shape*#features*#agent]
        agent_qs = agent_qs.reshape(-1, 1, self.n_agents*self.action_shape)  # [#batch*#sequence, 1, #agent*#action_shape]
        # First layer
        w1 = self.hyper_w_1(states).abs() if self.abs else self.hyper_w_1(states)  # [#batch*#sequence, action_shape*self.embed_dim*#agent]
        b1 = self.hyper_b_1(states)  # [#batch*#sequence, self.embed_dim]
        w1 = w1.view(-1, self.n_agents*self.action_shape, self.embed_dim)  # [#batch*#sequence, #agent*action_shape, self.embed_dim]
        b1 = b1.view(-1, 1, self.embed_dim)   # [#batch*#sequence, 1, self.embed_dim]
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)  # [#batch*#sequence, 1, self.embed_dim]

        # Second layer
        w_final = self.hyper_w_final(states).abs() if self.abs else self.hyper_w_final(states)  # [#batch*#sequence, self.embed_dim]
        w_final = w_final.view(-1, self.embed_dim, 1)  # [#batch*#sequence, self.embed_dim, 1]
        # State-dependent bias
        v = self.V(states).view(-1, 1, 1)  # [#batch*#sequence, 1, 1]
        # Compute final output
        y = torch.bmm(hidden, w_final) + v  
        # Reshape and return
        q_tot = y.view(bs, -1, 1) # [#batch, #sequence, 1]
        return q_tot

class QMix_Trainer():
    def __init__(self, replay_buffer, n_agents, obs_dim, state_dim, action_shape, action_dim, hidden_dim, hypernet_dim, target_update_interval, lr=5e-4, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995):
        self.replay_buffer = replay_buffer

        self.action_dim = action_dim
        self.action_shape = action_shape
        self.n_agents = n_agents
        self.target_update_interval = target_update_interval
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        self.agent = RNNAgent(obs_dim, action_shape,
                              action_dim, hidden_dim, self.epsilon).to(device)
        self.target_agent = RNNAgent(
            obs_dim, action_shape, action_dim, hidden_dim, epsilon=0.0).to(device)
        
        self.mixer = QMix(state_dim, n_agents, action_shape,
                          hidden_dim, hypernet_dim).to(device)
        self.target_mixer = QMix(state_dim, n_agents, action_shape,
                          hidden_dim, hypernet_dim).to(device)
        
        self._update_targets()
        self.update_cnt = 0
        
        self.criterion = nn.MSELoss()

        self.optimizer = optim.AdamW(
            list(self.agent.parameters())+list(self.mixer.parameters()), 
            lr=lr,
            weight_decay=0.001)

    def sample_action(self):
        probs = torch.FloatTensor(
            np.ones(self.action_dim)/self.action_dim).to(device)
        dist = Categorical(probs)
        action = dist.sample((self.n_agents, self.action_shape))

        return action.type(torch.FloatTensor).numpy()

    def get_action(self, state, hidden_in):
        '''
        @return:
            action: w/ shape [#active_as]
        '''

        action, hidden_out = self.agent.get_action(state, hidden_in)

        return action, hidden_out

    def push_replay_buffer(self, ini_hidden_in, ini_hidden_out, episode_observation, episode_state, episode_next_state, episode_action,
                           episode_reward, episode_next_observation):
        '''
        @brief: push arguments into replay buffer
        '''
        self.replay_buffer.push(ini_hidden_in, ini_hidden_out, episode_observation, episode_state, episode_next_state, episode_action,
                                episode_reward, episode_next_observation)

    def update(self, batch_size):
        current_loss = 100
        total_epoch = 0
        num_epoch = 10
        # 1. Lấy batch từ replay buffer
        hidden_in, hidden_out, observation, state, next_state, action, reward, next_observation = self.replay_buffer.sample(
            batch_size)

        # Converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor.
        observation = np.array(observation)
        state = np.array(state)
        next_state = np.array(next_state)
        next_observation = np.array(next_observation)
        action = np.array(action)
        reward = np.array(reward)
        
        observation = torch.FloatTensor(observation).to(device) # [#batch, sequence, #agents, #features*action_shape]
        state = torch.FloatTensor(state).to(device) #
        next_state = torch.FloatTensor(next_state).to(device)
        next_observation = torch.FloatTensor(next_observation).to(device)
        action = torch.LongTensor(action).to(device) # [#batch, sequence, #agents, #action_shape]
        reward = torch.FloatTensor(reward).unsqueeze(-1).to(device) # reward is scalar, add 1 dim to be [reward] at the same dim

        while(current_loss > 0.1 and total_epoch < 10):
            for epoch in range(1, num_epoch + 1):
                # 2. Tính current Q values
                agent_outs, _ = self.agent(observation, hidden_in) # [#batch, #sequence, #agent, action_shape, num_actions]
                chosen_action_qvals = torch.gather(  # [#batch, #sequence, #agent, action_shape]
                    agent_outs, dim=-1, index=action.unsqueeze(-1)).squeeze(-1)
                qtot = self.mixer(chosen_action_qvals, state) # [#batch, #sequence, 1]

                # 3. Tính target Q values
                target_agent_outs, _ = self.target_agent(next_observation, hidden_out)
                target_max_qvals = target_agent_outs.max(dim=-1, keepdim=True)[0] # [#batch, #sequence, #agents, action_shape]
                target_qtot = self.target_mixer(target_max_qvals, next_state)

                # 4. Tính reward và targets
                reward_epoch = _calc_reward(reward)
                targets = self._build_td0_targets(reward_epoch, target_qtot)

                # 5. Tính loss và update
                loss = self.criterion(qtot, targets.detach())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                current_loss = loss.item()
            
                if epoch % 10 == 0:
                    print(f'Epoch {epoch}/{total_epoch+1}, Loss: {current_loss}')
            self.update_cnt += 1
            if self.update_cnt % self.target_update_interval == 0:
                self._update_targets()
            total_epoch += 1
        # Decay epsilon after each update
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.agent.epsilon = self.epsilon

        return current_loss, reward_epoch.sum(dim=1)[0].cpu().item()

    def _build_td_lambda_targets(self, rewards, target_qs, gamma=0.99, td_lambda=0.6):
        '''
        @params:
            rewards: [#batch, #sequence, 1]
            target_qs: [#batch, #sequence, 1]
        '''
        ret = target_qs.new_zeros(*target_qs.shape)
        ret[:, -1] = target_qs[:, -1]
        # backwards recursive update of the "forward view"
        for t in range(ret.shape[1] - 2, -1, -1):
            ret[:, t] = td_lambda * gamma * ret[:, t+1] + (rewards[:, t] + (1 - td_lambda) * gamma * target_qs[:, t+1])
        return ret

    def _build_td0_targets(self, rewards, target_qs, gamma=0.99):
        """
        Tính toán target Q-values theo công thức Q-learning: Q(s,a) = r + γ max_a' Q(s',a')
        
        @params:
            rewards: [#batch, #sequence, 1] - Phần thưởng tức thời
            target_qs: [#batch, #sequence, 1] - Q values từ target network
        @return:
            ret: [#batch, #sequence, 1] - Target Q values
        """
        ret = target_qs.new_zeros(*target_qs.shape)
        ret[:, -1] = rewards[:, -1]
        # Q(s,a) = r + γ max_a' Q(s',a')
        for t in range(ret.shape[1] - 2, -1, -1):
            ret[:, t] = rewards[:, t] + gamma * target_qs[:, t+1]
        return ret

    def _update_targets(self):
        for target_param, param in zip(self.target_mixer.parameters(), self.mixer.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_agent.parameters(), self.agent.parameters()):
            target_param.data.copy_(param.data)

    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        torch.save(self.agent.state_dict(), path+'_agent')
        torch.save(self.mixer.state_dict(), path+'_mixer')

    def load_model(self, path, map_location):
        self.agent.load_state_dict(torch.load(path+'_agent', map_location=map_location, weights_only=True))
        self.mixer.load_state_dict(torch.load(path+'_mixer', map_location=map_location, weights_only=True))

        self.agent.eval()
        self.mixer.eval()
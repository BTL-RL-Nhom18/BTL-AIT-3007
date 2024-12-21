import torch
import numpy as np

def _calc_reward(rewards):
    """
    Tính toán rewards
    rewards shape: [batch, seq, n_agents, 1]
    """
    env_rewards = rewards.clone().squeeze(-1)
 
    # # Tính mean reward cho các agent còn sống
    # alive_mask = (env_rewards != 0).float()
    # num_alive = alive_mask.sum(dim=2, keepdim=True)
    # num_alive = torch.clamp(num_alive, min=1.0)
    # env_rewards = (env_rewards * alive_mask).sum(dim=2, keepdim=True) / num_alive
    
    # return env_rewards # [batch, sequence, 1]
    return env_rewards.sum(dim=2, keepdim=True) / 81 # [batch, sequence, 1]

# if __name__ == "__main__":
#     rewards = torch.rand(1, 1000, 81, 1)
#     state = torch.randint(0, 2, (1, 1000, 45, 45, 5))
#     print(_calc_reward(rewards=rewards, state=state).shape)
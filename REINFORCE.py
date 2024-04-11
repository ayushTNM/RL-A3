import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import gymnasium as gym

# Define the policy network
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=32):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        # self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.mean = nn.Linear(hidden_size, action_dim)
        self.std = nn.Linear(hidden_size, action_dim)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        # x = torch.tanh(self.fc2(x))
        mean = self.mean(x)
        std = torch.exp(self.std(x)) # To keep positive
        return mean, std

# REINFORCE algorithm with entropy regularization
def reinforce(env_name, num_episodes=1000, lr=0.01, gamma=0.99, entropy_coef=0.01):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    policy_network = PolicyNetwork(state_dim, action_dim)
    optimizer = optim.Adam(policy_network.parameters(), lr=lr)

    progrss_bar = tqdm(range(num_episodes))

    for episode in progrss_bar:
        log_probs = []
        rewards = []

        # for _ in range(batch_size):
        state, _ = env.reset()
        done = False
        episode_reward = 0

        while not done:
            state[2] /= 8   # normalize anfular velocity
            mean, std = policy_network(torch.FloatTensor(state))
            action_dist = torch.distributions.Normal(mean, std)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            action = torch.clamp(action, env.action_space.low[0], env.action_space.high[0])
            # print(action)
            next_state, reward, term, trunc, _ = env.step(action.detach().numpy())
            done = term or trunc
            log_probs.append(log_prob)
            rewards.append(reward)
            state = next_state
            episode_reward += reward

        # Calculate discounted rewards
        discounted_rewards = []
        R = 0
        for r in rewards[::-1]:
            R = r + gamma * R
            discounted_rewards.insert(0, R)

        # Normalize discounted rewards
        discounted_rewards = torch.FloatTensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

        # Calculate loss
        loss = 0
        for log_prob, R in zip(log_probs, discounted_rewards):
            loss -= (log_prob * R)

        # Add entropy regularization
        entropy = torch.mean(0.5 * torch.log(2 * np.pi * std**2)) # Compute the entropy of the Gaussian action distribution
        loss += entropy_coef * entropy

        # Update policy network
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print episode statistics
        if episode % 10 == 0:
            print(f"Episode {episode}, Reward: {episode_reward}")

    env.close()

# Run REINFORCE algorithm
reinforce("Pendulum-v1")

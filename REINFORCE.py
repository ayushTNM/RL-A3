import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import gymnasium as gym
import matplotlib.pyplot as plt
from gymnasium.wrappers import RecordEpisodeStatistics, NormalizeObservation

def create_env(env_name, rm = None):
    env = gym.make(env_name, render_mode = rm)
    env = RecordEpisodeStatistics(env)
    env = NormalizeObservation(env)         # Might remove later if not explicitly needed
    return env

# Define the policy network
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.mean_layer = nn.Linear(hidden_size, action_dim)
        self.std_layer = nn.Linear(hidden_size, action_dim)
        
        # Initialization using Xavier
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.mean_layer.weight)
        nn.init.xavier_uniform_(self.std_layer.weight)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        mean = self.mean_layer(x)
        std = torch.exp(self.std_layer(x)) # To keep positive
        return mean, std

# REINFORCE algorithm with entropy regularization
def reinforce(env_name, policy = None, num_timesteps=200_000, lr=1e-3, gamma=0.92, entropy_coef=0.01, eval_interval = 2000):
    env = create_env(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    if policy == None:
        policy = PolicyNetwork(state_dim, action_dim)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    progress_bar = tqdm(range(num_timesteps))

    state, info = env.reset()
    episode = 1
    log_probs = []
    rewards = []
    entropies = []
    done = False
    eval_timesteps, eval_returns = [], []
    for ts in progress_bar:
        if ts % eval_interval == 0:
            eval_returns.append(evaluate(env_name, policy))
            eval_timesteps.append(ts)
            
        mean, std = policy(torch.FloatTensor(state))
        action_dist = torch.distributions.Normal(mean, std)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        action = torch.clamp(action, env.action_space.low[0], env.action_space.high[0])
        # print(action)
        next_state, reward, term, trunc, info = env.step(action.detach().numpy())
        log_probs.append(log_prob)
        rewards.append(reward)
        entropies.append(action_dist.entropy())
        done = term or trunc
        if done:
            episode+=1
            
            progress_bar.desc = f"episode: {episode}, rew. {info['episode']['r'][0]}"
            
            # Calculate discounted rewards
            Qs = []
            Q = 0
            for r in rewards[::-1]:
                Q = r + gamma * Q
                Qs.insert(0, Q)
            
            # Calculate loss
            loss = - torch.tensor(log_probs) * torch.tensor(Qs)
                        
            # Add entropy regularization
            # entropy = torch.mean(0.5 * torch.log(2 * np.pi * std**2)) # Compute the entropy of the Gaussian action distribution
            loss = torch.sum(loss + (entropy_coef * torch.tensor(entropies)))

            # Update policy network
            optimizer.zero_grad()
            # print(loss)
            loss.requires_grad = True
            loss.backward()
            optimizer.step()
            
            state, _ = env.reset()
            log_probs = []
            rewards = []
            entropies = []
            done = False
        else:
            state = next_state

    env.close()
    return policy, eval_returns, eval_timesteps
    
def playout(policy, env_name):
    env = create_env(env_name, rm="rgb_array")
    # env.action_space.seed(42)
    env = gym.wrappers.RecordVideo(env=env, video_folder='./', name_prefix="reinforce")

    state, _ = env.reset()

    env.start_video_recorder()
    
    for _ in range(1000):
        mean, std = policy(torch.FloatTensor(state))
        action_dist = torch.distributions.Normal(mean, std)
        action = action_dist.sample()
        action = torch.clamp(action, env.action_space.low[0], env.action_space.high[0])
        # print(action)
        next_state, reward, term, trunc, _ = env.step(action.detach().numpy())

        if term or trunc:
            break
        else:
            state = next_state
    env.close_video_recorder()
    env.close()
    
    
def evaluate(env_name, policy, num_episodes=100, comment=None):
    env = create_env(env_name)
    res = []
    for _ in range(num_episodes):
        state = torch.tensor(env.reset()[0])
        term, trunc = False, False
        total_reward = 0

        while not term and not trunc:
            mean, std = policy(torch.FloatTensor(state))
            action_dist = torch.distributions.Normal(mean, std)
            action = action_dist.sample()
            action = torch.clamp(action, env.action_space.low[0], env.action_space.high[0])
            # print(action)
            next_state, reward, term, trunc, _ = env.step(action.detach().numpy())
            next_state = torch.tensor(next_state)
            total_reward += reward
            state = next_state
        res.append(total_reward)
    env.close()
    
    if comment is not None:
        print(comment, f'For {num_episodes} episodes, mean reward {np.mean(res)}, std {np.std(res)}', sep=', ')
    
    return res

# Run REINFORCE algorithm
env_name = "Pendulum-v1"
env = create_env(env_name)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
policy = PolicyNetwork(state_dim, action_dim)
evaluate(env_name, policy, comment="Random Policy")

policy, rets, stps = reinforce(env_name)
evaluate(env_name, policy, comment="Policy after approx. 200k env steps")

# Plotting
plt.plot(stps, np.mean(rets,axis=1))
plt.xlabel('Timesteps')
plt.ylabel('Returns')
plt.title('Returns vs Timesteps')
plt.grid(True)
plt.savefig("test.pdf")
# plt.show()

playout(policy, env_name)

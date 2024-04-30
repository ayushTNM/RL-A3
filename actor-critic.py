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
    def __init__(self, state_dim, action_dim, lr = 1e-3, hidden_size=64):
        super(PolicyNetwork, self).__init__()
        self.input = nn.Linear(state_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, hidden_size)
        self.linear5 = nn.Linear(hidden_size, hidden_size)
        self.linear6 = nn.Linear(hidden_size, hidden_size)
        self.mean_layer = nn.Linear(hidden_size, action_dim)
        self.std_layer = nn.Linear(hidden_size, action_dim)
        
        self.optim = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        x = torch.relu(self.input(x))
        x = torch.relu(self.linear2(x))
        x = torch.relu(self.linear3(x))
        x = torch.relu(self.linear4(x))
        x = torch.relu(self.linear5(x))
        x = torch.relu(self.linear6(x))
        mean = self.mean_layer(x)
        std = torch.exp(self.std_layer(x)) # To keep positive
        return mean, std
    
    def backprop(self, loss):
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        
class ValueNetwork(nn.Module):
    def __init__(self, input_dim, lr = 1e-3, hidden_size=64):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, hidden_size)
        self.linear5 = nn.Linear(hidden_size, hidden_size)
        self.linear6 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, 1)
        
        self.optim = optim.Adam(self.parameters(), lr=lr)
        self.loss= nn.MSELoss()

    def forward(self, x):
        # Forward pass through the layers with sigmoid activation
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = torch.relu(self.linear3(x))
        x = torch.relu(self.linear4(x))
        x = torch.relu(self.linear5(x))
        x = torch.relu(self.linear6(x))
        x = self.output(x)
        return x
    
    def backprop(self, loss):
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

# actor-critic algorithm with entropy regularization
def actor_critic(env_name, num_timesteps=200_000, n = 30, pol_lr=1e-3, val_lr=1e-3, gamma=0.92, entropy_coef=0.01, eval_interval = 2000):
    env = create_env(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    policy_network = PolicyNetwork(state_dim, action_dim, lr = pol_lr)
    value_network = ValueNetwork(state_dim, lr = val_lr)
    eval_timesteps, eval_returns = [], []

    progress_bar = tqdm(range(num_timesteps))
    
    bootstrap = True
    baseline_substraction = True

    state, info = env.reset()
    states, actions, log_probs, rewards, entropies = [], [], [], [], []
    done = False
    episode = 1

    for ts in progress_bar:
        
        if ts % eval_interval == 0:
            eval_returns.append(evaluate(env_name, policy_network))
            eval_timesteps.append(ts)

        mean, std = policy_network(torch.FloatTensor(state))
        action_dist = torch.distributions.Normal(mean, std)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        action = torch.clamp(torch.tanh(action), env.action_space.low[0],env.action_space.high[0])
        next_state, reward, term, trunc, info = env.step(action.detach().numpy())
        done = term or trunc
        states.append(torch.FloatTensor(next_state))
        actions.append(action)
        entropies.append(action_dist.entropy())
        log_probs.append(log_prob)
        rewards.append(reward)

        if done:
            episode+=1
            # Print episode statistics
            progress_bar.desc = f"episode: {episode}, rew. {info['episode']['r'][0]}"
            
            values = value_network(torch.stack(states))
            Qs = rewards * gamma**np.arange(len(rewards))    # Discounted rewards
            
            emperical_Qs = []
            for ind_Q in range(len(Qs)):
                emperical_Qs.append(np.sum(Qs[ind_Q:]))
            
            with torch.no_grad():
                if bootstrap:
                    for t in range(len(rewards)):
                        if t + n < len(rewards):
                            Qs[t] = np.sum(Qs[t:t+n]) + (values[t+n].item() * gamma**(t+n))
                        else:
                            Qs[t] = np.sum(Qs[t:t+n])
                            
            value_loss = torch.sum(value_network.loss(torch.FloatTensor(Qs).unsqueeze(1), values))
            
            with torch.no_grad():
                if baseline_substraction:
                    Qs = Qs - values.squeeze().numpy()
                    
            policy_loss = - torch.tensor(log_probs) * Qs
            policy_loss = torch.sum(policy_loss + (entropy_coef * torch.tensor(entropies)))  # Add entropy regularization term

            
            # Descent value loss
            value_network.backprop(value_loss)

            print(np.mean(emperical_Qs),np.mean((np.array(emperical_Qs) - value_network(torch.stack(states)).squeeze().detach().numpy())))
            
            # Ascent policy gradient
            policy_loss.requires_grad = True
            policy_network.backprop(policy_loss)
            
            state, info = env.reset()
            states, actions, log_probs, rewards, entropies = [], [], [], [], []
            done = False
        else:
            state = next_state
    env.close()
    
    return policy_network, eval_returns, eval_timesteps
    
def playout(policy, env_name):
    env = create_env(env_name, "rgb_array")
    # env.action_space.seed(42)
    env = gym.wrappers.RecordVideo(env=env, video_folder='./', name_prefix="act-crit")

    state, _ = env.reset()

    env.start_video_recorder()
    
    for _ in range(1000):
        mean, std = policy(torch.FloatTensor(state))
        action_dist = torch.distributions.Normal(mean, std)
        action = torch.clamp(torch.tanh(action_dist.sample()), env.action_space.low[0],env.action_space.high[0])

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
        state, info = env.reset()
        term, trunc = False, False

        while not term and not trunc:
            mean, std = policy(torch.FloatTensor(state))
            action_dist = torch.distributions.Normal(mean, std)
            action = torch.clamp(torch.tanh(action_dist.sample()), env.action_space.low[0],env.action_space.high[0])

            next_state, _, term, trunc, info = env.step(action.detach().numpy())
            state = next_state
        res.append(info['episode']['r'][0])
    env.close()
    
    if comment is not None:
        print(comment, f'For {num_episodes} episodes, mean reward {np.mean(res)}, std {np.std(res)}', sep=', ')
    
    return res

# Run actor-critic algorithm
env_name = "Pendulum-v1"
env = create_env(env_name)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
policy = PolicyNetwork(state_dim, action_dim)
evaluate(env_name,policy, comment="Random Policy")

policy, rets, stps = actor_critic(env_name)
evaluate(env_name,policy, comment="Policy after approx. 200k env steps")

# Plotting
plt.plot(stps, np.mean(rets,axis=1))
plt.xlabel('Timesteps')
plt.ylabel('Returns')
plt.title('Returns vs Timesteps')
plt.grid(True)
plt.savefig("test.pdf")
# plt.show()

playout(policy, env_name)
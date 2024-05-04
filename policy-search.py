import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import gymnasium as gym
import matplotlib.pyplot as plt
from gymnasium.wrappers import RecordEpisodeStatistics

def create_env(env_name, rm = None):
    env = gym.make(gym_name(env_name), render_mode = rm)
    env = RecordEpisodeStatistics(env)
    
    if env_name == "Pendulum-discrete":
        env = pendulum_discrete_wrapper(env)
    return env

def gym_name(env_name):
    return env_name if env_name != "Pendulum-discrete" else "Pendulum-v1"

class Wrapper:
    def __init__(self, inner):
        self.inner = inner
    
    def __getattr__(self, name):
        return getattr(self.inner, name)

def pendulum_discrete_wrapper(env):
    env = Wrapper(env)
    low = env.action_space.low[0]
    high = env.action_space.high[0]
    env.action_space = gym.spaces.Discrete(3)
    env.step = lambda action: env.inner.step(low + np.array([action]) / 2 * (high - low))
    return env

# Define the policy network
class PolicyNetwork(nn.Module):
    def __init__(self, state_space, action_space, lr = None, hidden_count = 2, hidden_size=64):
        super(PolicyNetwork, self).__init__()
        
        self.is_discrete = isinstance(action_space, gym.spaces.Discrete)
        
        self.state_space = state_space
        self.action_space = action_space
        
        state_dim = self.state_space.shape[0]
        action_dim = self.action_space.n if self.is_discrete else self.action_space.shape[0]
        
        # Input layer
        self.input_layer = nn.Linear(state_dim, hidden_size)
        
        # Hidden layers  with relu activations
        self.hidden_layers = nn.Sequential(*[nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU()) for _ in range(hidden_count)])
            
        # Output layer
        self.mean_layer = nn.Linear(hidden_size, action_dim)
        self.std_layer = nn.Linear(hidden_size, action_dim)
        
        if not lr is None:
            self.optim = optim.Adam(self.parameters(), lr=lr)
        
        self.log_probs = []
        self.entropies = []

    def forward(self, x):
        # Forward pass through the layers
        x = torch.relu(self.input_layer(x))
        x = self.hidden_layers(x)
        
        if self.is_discrete:
            out = torch.softmax(self.mean_layer(x), dim=-1)
            action_distr = torch.distributions.Categorical(out)
        else:
            mean = self.mean_layer(x)
            std = torch.exp(self.std_layer(x)) # To keep positive
            action_distr = torch.distributions.Normal(mean, std)
        
        action = action_distr.sample()
        log_prob = action_distr.log_prob(action)
        if not self.is_discrete:
            action = self.action_space.low[0] + (torch.tanh(action) + 1) / 2 * (self.action_space.high[0] - self.action_space.low[0])
        
        return action.detach().unsqueeze(0).numpy()[0], log_prob, action_distr.entropy()
    
    def backprop(self, loss):
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        
        
class ValueNetwork(nn.Module):
    def __init__(self, state_space, lr = None, hidden_count = 2, hidden_size = 64):
        super().__init__()
        state_dim = state_space.shape[0]
        
        # Input layer
        self.input_layer = nn.Linear(state_dim, hidden_size)
        
        # Hidden layers with relu activations
        self.hidden_layers = nn.Sequential(*[nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU()) for _ in range(hidden_count)])
        
        # Output layer
        self.output_layer = nn.Linear(hidden_size, 1)
        
        if not lr is None:
            self.optim = optim.Adam(self.parameters(), lr=lr)
            self.loss= nn.MSELoss()

    def forward(self, x):
        # Forward pass through the layers
        x = torch.relu(self.input_layer(x))
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x

    def backprop(self, loss):
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

# Base policy search algorithm with entropy regularization
def policy_search(env_name, num_timesteps=200_000, n = 30, pol_lr=1e-3, val_lr=1e-3, gamma=0.92, entropy_coef=0.01, eval_interval = 5000, bootstrap=True, baseline_substraction=True):
    env = create_env(env_name)
    
    policy_net = PolicyNetwork(env.observation_space, env.action_space, lr = pol_lr)
    value_net = ValueNetwork(env.observation_space, lr = val_lr)
    eval_timesteps, eval_returns = [], []
    log_probs = []


    progress_bar = tqdm(range(num_timesteps))
    state, info = env.reset()
    states, rewards, log_probs, entropies = [], [], [], []
    done = False
    episode = 1

    for ts in progress_bar:
        
        if eval_interval is not None and ts % eval_interval == 0:
            eval_returns.append(evaluate(env_name, policy_net))
            eval_timesteps.append(ts)

        action, log_prob, entropy = policy_net(torch.FloatTensor(state))
        
        next_state, reward, term, trunc, info = env.step(action)
        done = term or trunc
        states.append(torch.FloatTensor(next_state))
        entropies.append(entropy)
        log_probs.append(log_prob)
        rewards.append(reward)

        if done:
            episode+=1
            
            # Print episode statistics
            progress_bar.desc = f"episode: {episode}, rew. {info['episode']['r'][0]:.5}"
            
            values = value_net(torch.stack(states))
            
            emperical_Qs = []
            for ind_Q in range(len(rewards)):
                emperical_Qs.append(np.sum(rewards[ind_Q:] * gamma**np.arange(len(rewards) - ind_Q)))
            
            Qs = np.array(emperical_Qs)
            if bootstrap:
                with torch.no_grad():
                    for t in range(len(rewards)):
                        if t + n < len(rewards):
                            Qs[t] = np.sum(rewards[t:t+n] * gamma**np.arange(n)) + (values[t+n].item() * gamma**n)
                        else:
                            Qs[t] = np.sum(rewards[t:] * gamma**np.arange(len(rewards) - t))
                    
            if baseline_substraction:
                with torch.no_grad():
                    advantages = Qs - values.squeeze().numpy()
            else:
                advantages = Qs
                    
            # if neither: reinforce
            if bootstrap or baseline_substraction:
                # Descent value gradient
                value_loss = torch.sum(value_net.loss(torch.FloatTensor(Qs).unsqueeze(1), values))
                value_net.backprop(value_loss)            
                
                emp_abs_qvalue = np.mean(np.abs(emperical_Qs))
                mean_abs_diff_qvalue = np.mean(np.abs(np.array(emperical_Qs) - value_net(torch.stack(states)).squeeze().detach().numpy()))
                progress_bar.desc += f", qvalue model/emp error {(mean_abs_diff_qvalue / emp_abs_qvalue):.5}"

            # Ascent policy gradient
            policy_loss = - torch.stack(log_probs) * torch.tensor(advantages)
            policy_loss = torch.sum(policy_loss + (entropy_coef * torch.stack(entropies)))  # Add entropy regularization term
            policy_net.backprop(policy_loss)
            
            state, info = env.reset()
            states, rewards, log_probs, entropies = [], [], [], []
            done = False
        else:
            state = next_state
    env.close()
    
    return policy_net, eval_returns, eval_timesteps

    
def playout(env_name, policy_net, file_prefix, record = True):
    env = create_env(env_name, "rgb_array")

    state, _ = env.reset()
    term, trunc = False, False

    if record:
        env = gym.wrappers.RecordVideo(env=env, video_folder='./', name_prefix=file_prefix)
        env.start_video_recorder()
    
    while not term and not trunc:
        action,_,_ = policy_net(torch.FloatTensor(state))
        next_state, _, term, trunc, _ = env.step(action)
        state = next_state
        
    if record:
        env.close_video_recorder()
    env.close()
    
    
def evaluate(env_name, policy_net = None, num_episodes=10, comment=None):
    env = create_env(env_name)
    if policy_net is None:
        policy_net = PolicyNetwork(env.observation_space, env.action_space)
    res = []
    for _ in range(num_episodes):
        state, info = env.reset()
        term, trunc = False, False

        while not term and not trunc:
            action,_,_ = policy_net(torch.FloatTensor(state))
            next_state, _, term, trunc, info = env.step(action)
            state = next_state
        res.append(info['episode']['r'][0])
    env.close()
    
    if comment is not None:
        print(comment, f'For {num_episodes} episodes, mean reward {np.mean(res)}, std {np.std(res)}', sep=', ')
    
    return res

def reinforce(env_name, num_timesteps=200_000, n = 30, pol_lr=1e-3, val_lr=1e-3, gamma=0.92, entropy_coef=0.01, eval_interval = 5000):
    return policy_search(env_name, num_timesteps, n, pol_lr, val_lr, gamma, entropy_coef, eval_interval, bootstrap=False, baseline_substraction=False)

def actor_critic(env_name, num_timesteps=200_000, n = 30, pol_lr=1e-3, val_lr=1e-3, gamma=0.92, entropy_coef=0.01, eval_interval = 5000, bootstrap=True, baseline_substraction=True):
    return policy_search(env_name, num_timesteps, n, pol_lr, val_lr, gamma, entropy_coef, eval_interval, bootstrap, baseline_substraction)

if __name__ == "__main__":
    env_name = "Pendulum-discrete" # "LunarLander-v2", "Pendulum-v1", "Pendulum-discrete"
    evaluate(env_name, comment="Random Policy")
    
    num_timesteps = 1_000_000
    
    # Reinforce
    policy, rets, stps = reinforce(env_name, num_timesteps=num_timesteps, eval_interval=40_000)
    evaluate(env_name,policy, comment=f"Policy after {num_timesteps} evaluation steps")

    # Plotting
    plt.plot(stps, np.mean(rets,axis=1))
    plt.xlabel('Timesteps')
    plt.ylabel('Returns')
    plt.title('Returns vs Timesteps')
    plt.grid(True)
    plt.savefig("reinforce.pdf")

    playout(env_name, policy, "reinforce")
    
    # Actor-critic
    policy, rets, stps = actor_critic(env_name, num_timesteps=num_timesteps, eval_interval=40_000, bootstrap=True, baseline_substraction=True)
    evaluate(env_name,policy, comment=f"Policy after {num_timesteps} evaluation steps")

    # Plotting
    plt.plot(stps, np.mean(rets,axis=1))
    plt.xlabel('Timesteps')
    plt.ylabel('Returns')
    plt.title('Returns vs Timesteps')
    plt.grid(True)
    plt.savefig("actor-critic.pdf")

    playout(env_name, policy, "actor-critic")
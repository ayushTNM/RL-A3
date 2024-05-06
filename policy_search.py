import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import gymnasium as gym
import matplotlib.pyplot as plt
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo

# Wrapper for creating environments
def create_env(env_name, rm = None):
    env = gym.make(gym_name(env_name), render_mode = rm)
    env = RecordEpisodeStatistics(env)
    
    # call custom wrapper for pendulum-discrete env
    if env_name == "Pendulum-discrete":
        env = pendulum_discrete_wrapper(env)
    return env

# Fucntion to make sure Pendulum-discrete env can be called using make
def gym_name(env_name):
    return env_name if env_name != "Pendulum-discrete" else "Pendulum-v1"

# Simple wrapper class for custom Pendulum-discrete env
class Wrapper:
    def __init__(self, inner):
        self.inner = inner
    
    def __getattr__(self, name):
        return getattr(self.inner, name)

# Wrapper function to convert Pendulum-v1 to a discrete env
def pendulum_discrete_wrapper(env):
    env = Wrapper(env)
    low = env.action_space.low[0]
    high = env.action_space.high[0]
    env.action_space = gym.spaces.Discrete(3)
    env.step = lambda action: env.inner.step(low + np.array([action]) / 2 * (high - low))
    return env

# Define the policy network
class PolicyNetwork(nn.Module):
    def __init__(self, state_space, action_space, lr = None, hidden_count = 2, hidden_size=64, init = None):
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
            
        # Output layers
        self.mean_layer = nn.Linear(hidden_size, action_dim)
        self.std_layer = nn.Linear(hidden_size, action_dim)
        
        # Initialize optimizer if needed
        if not lr is None:
            self.optim = optim.Adam(self.parameters(), lr=lr)
            
        # Initialization using Xavier if specified else uniform
        if init == 'xavier':
            nn.init.xavier_uniform_(self.input_layer.weight)
            for layer in self.hidden_layers:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
            nn.init.xavier_uniform_(self.mean_layer.weight)
            nn.init.xavier_uniform_(self.std_layer.weight)

    def forward(self, x):
        # Forward pass through the layers
        x = torch.relu(self.input_layer(x))
        x = self.hidden_layers(x)
        
        # Adjust output layer based on if env is discrete
        if self.is_discrete:
            out = torch.softmax(self.mean_layer(x), dim=-1)
            action_distr = torch.distributions.Categorical(out)
        else:
            mean = self.mean_layer(x)
            std = nn.functional.softplus(self.std_layer(x)) # To keep positive
            action_distr = torch.distributions.Normal(mean, std)
        
        action = action_distr.sample()
        log_prob = action_distr.log_prob(action)
        
        # Scale to env action
        if not self.is_discrete:
            action = self.action_space.low[0] + (torch.tanh(action) + 1) / 2 * (self.action_space.high[0] - self.action_space.low[0])
        
        return action.detach().unsqueeze(0).numpy()[0], log_prob, action_distr.entropy()
    
    def backprop(self, loss):
        # Backpropagate
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        
# Define the value network
class ValueNetwork(nn.Module):
    def __init__(self, state_space, lr = None, hidden_count = 2, hidden_size = 64, init = None):
        super().__init__()
        state_dim = state_space.shape[0]
        
        # Input layer
        self.input_layer = nn.Linear(state_dim, hidden_size)
        
        # Hidden layers with relu activations
        self.hidden_layers = nn.Sequential(*[nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU()) for _ in range(hidden_count)])
        
        # Output layer
        self.output_layer = nn.Linear(hidden_size, 1)
        
        if not lr is None: # If reinforce
            self.optim = optim.Adam(self.parameters(), lr=lr)
            self.loss= nn.MSELoss()
            
        # Initialization using Xavier if specified else uniform
        if init == 'xavier':
            nn.init.xavier_uniform_(self.input_layer.weight)
            for layer in self.hidden_layers:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
            nn.init.xavier_uniform_(self.output_layer.weight)

    def forward(self, x):
        # Forward pass through the layers
        x = torch.relu(self.input_layer(x))
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x

    def backprop(self, loss):
        # Backpropagate
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

# Base policy search algorithm with entropy regularization
def policy_search(env_name, num_timesteps=200_000, n = 30, pol_lr=1e-3, val_lr=1e-3, gamma=0.92, entropy_coef=0.01, eval_interval = 5000, bootstrap=True, baseline_substraction=True, pol_init = None, val_init = None):
    env = create_env(env_name)
    
                                                                                # Define value network if not reinforce
    value_net = ValueNetwork(env.observation_space, lr = val_lr, init=val_init) if bootstrap or baseline_substraction else None
    policy_net = PolicyNetwork(env.observation_space, env.action_space, lr = pol_lr, init=pol_init)
    states, rewards, log_probs, entropies = [], [], [], []
    done = False
    
    eval_timesteps, eval_returns = [], []

    episode = 1
    state, _ = env.reset()
    progress_bar = tqdm(range(num_timesteps))
    
    for ts in progress_bar:
        # Store return of evaluation run every eval_interval timesteps
        if eval_interval is not None and ts % eval_interval == 0:
            eval_returns.append(evaluate(env_name, policy_net))
            eval_timesteps.append(ts)

        # Act
        action, log_prob, entropy = policy_net(torch.FloatTensor(state))
        next_state, reward, term, trunc, _ = env.step(action)
        
        done = term or trunc
        
        # Store info
        states.append(torch.FloatTensor(next_state))
        entropies.append(entropy)
        log_probs.append(log_prob)
        rewards.append(reward)

        # After each episode:
        if done:
            episode+=1
            
            # Print episode statistics
            progress_bar.desc = f"episode {episode}, rew. {env.return_queue[-1][-1]:.5}" # return_queue stores cummulative rewards/returns
            
            # Forward pass through value network if not reinforce
            if bootstrap or baseline_substraction:
                values = value_net(torch.stack(states))
            
            # Calculate emperical Q-values -> Monte Carlo
            emperical_Qs = []
            for ind_Q in range(len(rewards)):
                emperical_Qs.append(np.sum(rewards[ind_Q:] * gamma**np.arange(len(rewards) - ind_Q)))
            
            # Bootstrap if needed
            Qs = np.array(emperical_Qs)
            if bootstrap:
                with torch.no_grad():
                    for t in range(len(rewards)):
                        if t + n < len(rewards):
                            Qs[t] = np.sum(rewards[t:t+n] * gamma**np.arange(n)) + (values[t+n].item() * gamma**n)
                        else:
                            Qs[t] = np.sum(rewards[t:] * gamma**np.arange(len(rewards) - t))
            
            # Baseline substraction if needed
            advantages = Qs
            if baseline_substraction:
                with torch.no_grad():
                    advantages = Qs - values.squeeze().numpy()
                    
            # if not reinforce
            if bootstrap or baseline_substraction:
                # Descent value gradient
                value_loss = torch.sum(value_net.loss(torch.FloatTensor(Qs).unsqueeze(1), values))
                value_net.backprop(value_loss)            
                
                # Print ratio between Q-values predicted by value network and the empirical Q-values
                emp_abs_qvalue = np.mean(np.abs(emperical_Qs))
                mean_abs_diff_qvalue = np.mean(np.abs(np.array(emperical_Qs) - value_net(torch.stack(states)).squeeze().detach().numpy()))
                progress_bar.desc += f", qvalue model/emp error {(mean_abs_diff_qvalue / emp_abs_qvalue):.5}"

            # Ascent policy gradient
            policy_loss = - torch.stack(log_probs) * torch.tensor(advantages / np.std(advantages))
            policy_loss = torch.sum(policy_loss - (entropy_coef * torch.stack(entropies)))  # Add entropy regularization term
            policy_net.backprop(policy_loss)
            
            # Reset environment
            state, _ = env.reset()
            states, rewards, log_probs, entropies = [], [], [], []
            done = False
        else:
            state = next_state
    env.close()
    
    return policy_net, value_net, eval_returns, eval_timesteps

# Function for saving video of certain env for a given policy
def playout(env_name, policy_net, file_prefix, file_folder=''):
    # Initialize env with rgb_array render mode for saveing video
    env = create_env(env_name, "rgb_array")

    state, _ = env.reset()
    term, trunc = False, False

    # Initialize video recording
    env = RecordVideo(env=env, video_folder=file_folder, name_prefix=file_prefix)
    env.start_video_recorder()
    
    while not term and not trunc:
        action,_,_ = policy_net(torch.FloatTensor(state))
        next_state, _, term, trunc, _ = env.step(action)
        state = next_state
        
    # Stop video recording
    env.close_video_recorder()
    
    env.close()
    
# Evaluate/Test function
def evaluate(env_name, policy_net = None, num_episodes=10, comment=None):
    env = create_env(env_name)
    
    if policy_net is None:
        policy_net = PolicyNetwork(env.observation_space, env.action_space)
    
    res = []
    for _ in range(num_episodes):
        state, _ = env.reset()
        term, trunc = False, False

        while not term and not trunc:
            action,_,_ = policy_net(torch.FloatTensor(state))
            next_state, _, term, trunc, info = env.step(action)
            state = next_state
            
        # Append cummulative reward
        res.append(env.return_queue[-1][-1])
    env.close()
    
    # print stats
    if comment is not None:
        print(comment, f'For {num_episodes} episodes, mean reward {np.mean(res)}, std {np.std(res)}', sep=', ')
    
    return res

# Reinforce
def reinforce(env_name, num_timesteps=200_000, pol_lr=1e-3, gamma=0.92, entropy_coef=0.01, eval_interval = 5000, pol_init = None, val_init = None):
    return policy_search(env_name, num_timesteps, None, pol_lr, None, gamma, entropy_coef, eval_interval, bootstrap=False, baseline_substraction=False, pol_init=pol_init, val_init = val_init)

# Actor-critic
def actor_critic(env_name, num_timesteps=200_000, n = 30, pol_lr=1e-3, val_lr=1e-3, gamma=0.92, entropy_coef=0.01, eval_interval = 5000, bootstrap=True, baseline_substraction=True, pol_init = None, val_init = None):
    return policy_search(env_name, num_timesteps, n, pol_lr, val_lr, gamma, entropy_coef, eval_interval, bootstrap, baseline_substraction, pol_init = pol_init, val_init = val_init)

if __name__ == "__main__":
    env_name = "LunarLander-v2" # "LunarLander-v2", "Pendulum-v1", "Pendulum-discrete"
    
    # Baseline: Random policy
    evaluate(env_name, comment="Random Policy")
    
    num_timesteps = 1_000_001
    eval_interval = num_timesteps//100
    
    # # Reinforce
    # reinf_save_name = 'reinforce'
    # policy, _, rets, stps = reinforce(env_name, num_timesteps=num_timesteps, eval_interval=eval_interval, pol_init='xavier')
    # evaluate(env_name,policy, comment=f"Policy after {num_timesteps} evaluation steps")

    # # Plot reinforce
    # plt.plot(stps, np.mean(rets,axis=1))
    # plt.xlabel('Timesteps')
    # plt.ylabel('Returns')
    # plt.title('Returns vs Timesteps')
    # plt.grid(True)
    # plt.savefig(reinf_save_name+".pdf")

    # Save playout reinforce
    # playout(env_name, policy, file_prefix=reinf_save_name)
    
    # Actor-critic
    ac_save_name = "actor-critic"
    policy, value, rets, stps = actor_critic(env_name, num_timesteps=num_timesteps, eval_interval=eval_interval, bootstrap=True, baseline_substraction=True, pol_init='xavier', val_init='xavier')
    evaluate(env_name,policy, comment=f"Policy after {num_timesteps} evaluation steps")

    # Plot actor-critic
    plt.plot(stps, np.mean(rets,axis=1))
    plt.xlabel('Timesteps')
    plt.ylabel('Returns')
    plt.title('Returns vs Timesteps')
    plt.grid(True)
    plt.savefig(ac_save_name+".pdf")

    # Save playout actor-critic
    playout(env_name, policy, file_prefix=ac_save_name)
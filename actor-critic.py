import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal as torch_normal
import gymnasium as gym
import copy
import numpy as np
from tqdm import tqdm
import random as rn
import argparse

dev = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(dev)

class PolicyNet(nn.Module):
    def __init__(self, input_dim, output_dim, min_vals, max_vals, width=16, lr=1e-5, reg_term=1, track_loss_update=True):
        super().__init__()
        self.reg_term = reg_term
        self.track_loss_update = track_loss_update
        self.min_vals = torch.tensor(min_vals, dtype=torch.float32)
        self.max_vals = torch.tensor(max_vals, dtype=torch.float32)
        
        self.linear1 = nn.Linear(input_dim, width)
        self.linear2 = nn.Linear(width, width)
        self.output_mean = nn.Linear(width, output_dim)
        self.output_std = nn.Linear(width, output_dim)

        # Initialization using Xavier
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.xavier_uniform_(self.output_mean.weight)
        nn.init.xavier_uniform_(self.output_std.weight)
        
        # Define optimizer
        self.lr = lr
        self.optimizer = optim.SGD(self.parameters(), lr=lr)

    def get_dstb_params(self, inputs):
        # Get distribution parameters
        x = torch.sigmoid(self.linear1(inputs))
        x = torch.sigmoid(self.linear2(x))
        means = self.min_vals + torch.sigmoid(self.output_mean(x)) * (self.max_vals - self.min_vals)
        stds = (torch.sigmoid(self.output_std(x)) + 0.1) * (self.max_vals - self.min_vals)  # +0.1 temp hack to avoid mode collapse
        return means, stds

    def forward(self, inputs):
        # Forward pass through the layers with sigmoid activation
        means, stds = self.get_dstb_params(inputs)
        return torch.normal(means, stds)
    
    def update(self, _states, _actions, _rewards, qnet=None):
        if qnet is None:
            qnet = lambda *args: 0
        self.optimizer.zero_grad()
        
        def get_loss(loss_terms=None):
            loss = torch.tensor(0, dtype=torch.float32)
            loss.requires_grad_(True)
            
            loss_terms_precalc = loss_terms is not None
            
            if not loss_terms_precalc:
                loss_terms = []
            
            q_est_list = []
            for states, rewards in zip(_states, _rewards):
                rs = np.array(rewards)
                end_qvalue = qnet(states[-1])
                q_est_list.append(torch.tensor([np.sum(rs[i:]) + end_qvalue for i in range(len(states))]))
            
            q_est_mean = torch.mean(torch.tensor([list(t) for t in q_est_list]), 0)
            for states, actions, q_est in zip(_states, _actions, q_est_list):
                loss = loss - torch.sum((q_est - q_est_mean) * torch.stack([torch_normal(*self.get_dstb_params(state)).log_prob(action) for state, action in zip(states, actions)]))
                if not loss_terms_precalc:
                    loss_terms.append(-torch.sum((q_est - q_est_mean) * torch.stack([torch_normal(*self.get_dstb_params(state)).log_prob(action) for state, action in zip(states, actions)])))
            
            if not loss_terms_precalc:
                loss_terms = torch.tensor(loss_terms).detach()
            return loss, loss_terms
        
        loss, loss_terms = get_loss()
        loss.backward()
        self.optimizer.step()
        
        updated_loss, _ = get_loss(loss_terms) if self.track_loss_update else (None, None)
        
        return loss.item(), ((loss - updated_loss) / torch.std(loss_terms)).item() if self.track_loss_update else float('nan')

class QNet(nn.Module):
    def __init__(self, input_dim, width=16):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, width)
        self.linear2 = nn.Linear(width, width)
        self.output = nn.Linear(width, 1)

        # Initialization using Xavier
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.xavier_uniform_(self.output.weight)

    def forward(self, inputs):
        # Forward pass through the layers with sigmoid activation
        x = torch.sigmoid(self.linear1(inputs))
        x = torch.sigmoid(self.linear2(x))
        x = self.output(x)
        return x
    
def update_q_network(_states, _rewards, qnet, lr):
    qnet_optimizer = optim.Adam(qnet.parameters(), lr=lr)
    qnet_loss = torch.tensor(0.0)
    
    # Initialize total gradient with zeros matching the Q-network parameters
    # total_grad = [torch.zeros_like(param) for param in qnet.parameters()]
    
    for states, rewards in zip(_states, _rewards):
        # Estimate values of states using Q-network
        values = qnet(states[0])
        
        # Compute temporal difference errors
        td_errors = rewards[0] - values[0]
        
        # Compute gradients of squared TD errors with respect to Q-network parameters
        qnet.zero_grad()
        loss = torch.mean(td_errors ** 2)
        qnet_loss += loss.item()
        loss.backward()
        
    qnet_optimizer.step()
    
    return qnet_loss.item()



def actor_critic(n_timesteps, trace_sample_size=15, trace_depth=60, explore_steps=30, policy=None, live_playout=False):
    ref_env = gym.make('Pendulum-v1')

    if policy is None:
        policy = get_initial_policy()
    global _policy
    _policy = policy
    
    qnet = QNet(input_dim=3)
    state = torch.tensor(ref_env.reset()[0])
    
    curr_ep_reward = 0
    full_ep_reward = 0
    
    with tqdm(total=n_timesteps) as env_steps_bar:
        while env_steps_bar.n + trace_depth * trace_sample_size + explore_steps < n_timesteps:
            envs = [copy.deepcopy(ref_env) for _ in range(trace_sample_size)]
            _states = []
            _actions = []
            _rewards = []
            for env in envs:
                states, actions, rewards = playout(env, state, policy, trace_depth) 
                _states.append(states)
                _actions.append(actions)
                _rewards.append(rewards)
                env_steps_bar.update(trace_depth)
            for env in envs:
                env.close()     # try to avoid memory leaks
            policy_loss, improvement = policy.update(_states, _actions, _rewards, qnet)
            q_loss = update_q_network(_states, _rewards, qnet, policy.lr)
            state, rewards, ep_reset = env_advance(ref_env, state, policy, explore_steps)
            curr_ep_reward += rewards
            if ep_reset:
                full_ep_reward = curr_ep_reward
                curr_ep_reward = 0
            env_steps_bar.desc = f'Loss: {policy_loss:3f}, {q_loss:3f}, Improvement: {improvement:3f}, Last ep reward: {full_ep_reward:3f}'
    ref_env.close()
    
    if live_playout:
        live_play(policy, 'After training')
    
    return policy

def get_initial_policy():
    return PolicyNet(3, 1, [-2], [2])

def playout(env, initial_state, policy, trace_depth):
    states = [initial_state]
    actions = []
    rewards = []
    state = initial_state
    term, trunc = False, False
    for _ in range(trace_depth):
        if term or trunc:
            break
        action = policy(state).detach()
        next_state, reward, term, trunc, _ = env.step(action.numpy())
        state = torch.tensor(next_state)
        states.append(state)
        actions.append(action)
        rewards.append(reward)
    return states, actions, rewards

def env_advance(env, state, policy, explore_steps):
    term, trunc = False, False
    acc_rewards = 0
    for _ in range(explore_steps):
        if term or trunc:
            return torch.tensor(env.reset()[0]), acc_rewards, True
        action = policy(state).detach()
        next_state, reward, term, trunc, _ = env.step(action.numpy())
        acc_rewards += reward
        state = torch.tensor(next_state)
    return state, acc_rewards, False

def live_play(policy, comment, live=True):
    env = gym.make('Pendulum-v1', render_mode=('human' if live else None))
    state = torch.tensor(env.reset()[0])
    term, trunc = False, False
    total_reward = 0

    while not term and not trunc:
        action = policy(state).detach()
        next_state, reward, term, trunc, _ = env.step(action.numpy())
        next_state = torch.tensor(next_state)
        total_reward += reward
        state = next_state
    
    print(comment, f"Total reward of live episode: {total_reward}", sep=', ')
    env.close()

def evaluate(policy, num_episodes=10, comment=None):
    env = gym.make('Pendulum-v1')
    res = []
    for _ in range(num_episodes):
        state = torch.tensor(env.reset()[0])
        term, trunc = False, False
        total_reward = 0

        while not term and not trunc:
            action = policy(state).detach()
            next_state, reward, term, trunc, _ = env.step(action.numpy())
            next_state = torch.tensor(next_state)
            total_reward += reward
            state = next_state
        res.append(total_reward)
    env.close()
    
    if comment is not None:
        print(comment, f'For {num_episodes} episodes, mean reward {np.mean(res)}, std {np.std(res)}', sep=', ')
    
    return res

if __name__ == '__main__':
    policy = get_initial_policy()
    evaluate(policy, comment='Random policy')
    print()
    
    actor_critic(25000, policy=policy)
    evaluate(policy, comment='Policy after 25k env steps')
    print()
    
    actor_critic(75000, policy=policy)
    evaluate(policy, comment='Policy after 100k env steps')
    print()
    
    actor_critic(100000, policy=policy, live_playout=True)
    evaluate(policy, comment='Policy after 200k env steps')
    print()

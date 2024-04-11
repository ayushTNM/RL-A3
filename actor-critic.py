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
    def __init__(self, input_dim, output_dim, min_vals, max_vals, width=16, lr=1, reg_term=1):
        super().__init__()
        self.reg_term = reg_term
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
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

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
        loss = torch.tensor(0, dtype=torch.float32)
        loss.requires_grad_(True)
        
        loss_terms = []
        
        for states, actions, rewards in zip(_states, _actions, _rewards):
            rs = np.array(rewards)
            end_qvalue = qnet(states[-1])
            q_est = [np.sum(rs[i:]) + end_qvalue for i in range(len(states))]
            loss = loss - q_est[0] * torch.sum(torch.stack([torch_normal(*self.get_dstb_params(state)).log_prob(action) for state, action in zip(states, actions)])) # not exactly correct but should behave more stable without qnet or with unstable qnet
            loss_terms.append(-q_est[0] * torch.sum(torch.stack([torch_normal(*self.get_dstb_params(state)).log_prob(action) for state, action in zip(states, actions)])))
            # double check this code to make sure the tensors don't lose the gradient
            # loss = loss - self.reg_term * torch.sum(torch.stack([torch_normal(*self.get_dstb_params(state)).entropy() for state in states]))
        loss_terms = torch.tensor(loss_terms).detach()
        loss = (loss - torch.mean(loss_terms)) / torch.std(loss_terms)
        loss.backward()
        self.optimizer.step()
        return loss.item()

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

def actor_critic(n_timesteps, trace_sample_size=30, trace_depth=30, explore_steps=30):
    ref_env = gym.make('Pendulum-v1')
    policy = PolicyNet(3, 1, [-2], [2])
    qnet = None
    state = torch.tensor(ref_env.reset()[0])
    
    live_play(policy, 'Before training', False)
    
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
            loss = policy.update(_states, _actions, _rewards, qnet)
            env_steps_bar.desc = f'Loss: {loss:3f}'
            state = env_advance(ref_env, state, policy, explore_steps)
    ref_env.close()
    
    live_play(policy, 'After training')

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
        next_state, reward, term, trunc, _ = env.step(action)
        state = torch.tensor(next_state)
        states.append(state)
        actions.append(action)
        rewards.append(reward)
    return states, actions, rewards

def env_advance(env, state, policy, explore_steps):
    term, trunc = False, False
    for _ in range(explore_steps):
        if term or trunc:
            return torch.tensor(env.reset()[0])
        action = policy(state).detach()
        next_state, reward, term, trunc, _ = env.step(action)
        state = torch.tensor(next_state)
    return state

def live_play(policy, comment, live=True):
    env = gym.make('Pendulum-v1', render_mode=('human' if live else None))
    state = torch.tensor(env.reset()[0])
    term, trunc = False, False
    total_reward = 0

    while not term and not trunc:
        action = policy(state).detach().numpy()
        next_state, reward, term, trunc, _ = env.step(action)
        next_state = torch.tensor(next_state)
        total_reward += reward
        state = next_state
    
    print(comment, f"Total reward of live episode: {total_reward}", sep=', ')
    env.close()

if __name__ == '__main__':
    actor_critic(100000)

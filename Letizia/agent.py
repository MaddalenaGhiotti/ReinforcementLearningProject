import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal
import random

SEED = 42  # Set a seed for reproducibility

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def discount_rewards(r, gamma):
    discounted_r = torch.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size(-1))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64
        self.tanh = torch.nn.Tanh()

        """
            Actor network
        """
        self.fc1_actor = torch.nn.Linear(state_space, self.hidden)
        self.fc2_actor = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_actor_mean = torch.nn.Linear(self.hidden, action_space)
        
        # Learned standard deviation for exploration at training time 
        self.sigma_activation = F.softplus
        init_sigma = 0.5
        self.sigma = torch.nn.Parameter(torch.zeros(self.action_space)+init_sigma) 


        """
            Critic network
        """
        # TASK 3: critic network for actor-critic algorithm
        self.fc1_critic = torch.nn.Linear(state_space, self.hidden)
        self.fc2_critic = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_critic = torch.nn.Linear(self.hidden, 1)
       

        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)


    def forward(self, x):
        """
            Actor
        """
        x_actor = self.tanh(self.fc1_actor(x))
        x_actor = self.tanh(self.fc2_actor(x_actor))
        action_mean = self.fc3_actor_mean(x_actor)

        sigma = self.sigma_activation(self.sigma) 
        normal_dist = Normal(action_mean, sigma)


        """
            Critic
        """
        # TASK 3: forward in the critic network
        x_critic = self.tanh(self.fc1_critic(x))
        x_critic = self.tanh(self.fc2_critic(x_critic))
        action_value = self.fc3_critic(x_critic)

        
        return normal_dist, action_value


class Agent(object):
    def __init__(self, policy, device='cpu'):
        self.train_device = device
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
        

        self.gamma = 0.9
        self.I = 1 #gamma^t 
        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.rewards = []
        self.done = []


    def update_policy(self, algorithm):
        action_log_probs = torch.stack(self.action_log_probs, dim=0).to(self.train_device).squeeze(-1)
        states = torch.stack(self.states, dim=0).to(self.train_device).squeeze(-1)
        next_states = torch.stack(self.next_states, dim=0).to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        done = torch.Tensor(self.done).to(self.train_device)

        self.states, self.next_states, self.action_log_probs, self.rewards, self.done = [], [], [], [], []

        #
        # TASK 2:
        if algorithm == 'reinforce':
            #- compute discounted returns
            discounted_returns = discount_rewards(rewards, self.gamma)
            # - add fixed baseline
            baseline=20
            discounted_returns = discounted_returns - baseline
            discounted_returns = (discounted_returns - discounted_returns.mean()) / (discounted_returns.std() + 1e-8) # Normalize the returns

            #  - compute policy gradient loss function given actions and returns
            gradient_loss = -torch.mean(action_log_probs * discounted_returns)

            #  - compute gradients and step the optimizer
            self.optimizer.zero_grad()
            gradient_loss.backward()
            self.optimizer.step()


        #
        # TASK 3:
        if algorithm == 'actor-critic':
            #   - compute boostrapped discounted return estimates
            _, state_values = self.policy(states)                # V(s_t)
            _, next_state_values = self.policy(next_states)      # V(s_{t+1})


            done = done.float()
            td_target = rewards + self.gamma * next_state_values * (1 - done)  # if done=1 → no bootstrapping
            td_target = td_target.detach()  # Detach from the graph to avoid backpropagation through the next state value

            #   - compute advantage terms
            td_error = td_target - state_values  # delta = R_t + gamma*V(s_{t+1}) - V(s_t)

    
            actor_loss = -self.I*action_log_probs * td_error.detach()
            critic_loss = 1/2*(td_error.pow(2))  # MSE loss for critic

            #   - compute gradients and step the optimizer        
            self.optimizer.zero_grad()
            (actor_loss + critic_loss).backward()

            self.optimizer.step()

            self.I = self.I * self.gamma  # Update the I for the next step
        

            
        return  

    def reset_I(self):
        self.I = 1  # Reset the I for the next episode
        return     


    def get_action(self, state, evaluation=False):
        """ state -> action (3-d), action_log_densities """
            
        x = torch.from_numpy(state).float().to(self.train_device)

        normal_dist, _ = self.policy(x)

        if evaluation:  # Return mean
            return normal_dist.mean, None

        else:   # Sample from the distribution
            action = normal_dist.sample()

            # Compute Log probability of the action [ log(p(a[0] AND a[1] AND a[2])) = log(p(a[0])*p(a[1])*p(a[2])) = log(p(a[0])) + log(p(a[1])) + log(p(a[2])) ]
            action_log_prob = normal_dist.log_prob(action).sum()

            return action, action_log_prob


    def store_outcome(self, state, next_state, action_log_prob, reward, done):
        self.states.append(torch.from_numpy(state).float())
        self.next_states.append(torch.from_numpy(next_state).float())
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(torch.Tensor([reward]))
        self.done.append(done)


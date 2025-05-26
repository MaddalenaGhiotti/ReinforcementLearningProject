import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal

<<<<<<< HEAD
=======
##############
#######################
>>>>>>> 069e8ac (bastaaaa)
def discount_rewards(r, gamma):
    """
    Computation of return G for each  time-stamp and storage in a tensor
    @param r tensor of rewards
    @parm gamma discount factor
    """
    discounted_r = torch.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size(-1))):
        running_add = running_add * gamma + r[t]   #Computation of return G for each time-stamp
        discounted_r[t] = running_add   #Saving of returns in a tensor
    return discounted_r


class Policy(torch.nn.Module):  #Sub-class of NN PyTorch class
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space   #Attribute: state space
        self.action_space = action_space   #Attribute: action space
        self.hidden = 64   #Attribute: number of nodes in hidden layers
        self.tanh = torch.nn.Tanh()   #Attribute: activation function

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
        normal_dist = Normal(action_mean, sigma)   #Normalized policy outputs (probabilities of actions)


        """
            Critic
        """
        # TASK 3: forward in the critic network

        
        return normal_dist


class Agent(object):
    def __init__(self, policy, device='cpu', baseline=0):
        self.train_device = device
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)   #Optimization algorithm on the policy parameters

        self.gamma = 0.99   #Discount factor
        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.rewards = []
        self.done = []
        self.baseline = baseline


    def update_policy(self):
        """
        From trajectory to optimization step. Update policy params. All time-stamps simultaneously.
        """
        action_log_probs = torch.stack(self.action_log_probs, dim=0).to(self.train_device).squeeze(-1)   #Concatenate tensors in list along a new axis, move the new tensor on chosen devise and remove entra dimensions of size 1 from the end.
        states = torch.stack(self.states, dim=0).to(self.train_device).squeeze(-1)
        next_states = torch.stack(self.next_states, dim=0).to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        done = torch.Tensor(self.done).to(self.train_device)   #Create a 1-dimensional tensor from a list of bools

        self.states, self.next_states, self.action_log_probs, self.rewards, self.done = [], [], [], [], []
        
        '''
        MODIFIED
        '''
        returns = discount_rewards(rewards, self.gamma)
        returns -= returns.mean()
        returns/= returns.std()

        loss_fn =-torch.mean(action_log_probs * returns)
        #loss_fn = -(torch.from_numpy(self.gamma*np.ones((states.shape[0]))**np.arange(0,states.shape[0])).to(self.train_device)*(returns-self.baseline)*action_log_probs).sum()
        self.optimizer.zero_grad()
        loss_fn.backward()   #Compute the gradients of the loss w.r.t. each parameter
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(),1)
        self.optimizer.step()   #Compute a step of the optimization algorithm


        #
        # TASK 2:
        #   - compute discounted returns
        #   - compute policy gradient loss function given actions and returns
        #   - compute gradients and step the optimizer
        #


        #
        # TASK 3:
        #   - compute boostrapped discounted return estimates
        #   - compute advantage terms
        #   - compute actor loss and critic loss
        #   - compute gradients and step the optimizer
        #

        return        


    def get_action(self, state, evaluation=False):
        """ 
        Computation of one step of the trajectory
        state -> action (3-d), action_log_densities
        @param state current state
        @param evaluation do not act, return distribution mean
        @return action following action in trajectory
        @return action_log_prob probability of chosen action (joint probability of the three action values)
        """
        x = torch.from_numpy(state).float().to(self.train_device)

        normal_dist = self.policy(x)

        if evaluation:  # Return mean
            return normal_dist.mean, None

        else:   # Sample from the distribution  (choose an action)
            action = normal_dist.sample()

            # Compute Log probability of the action [ log(p(a[0] AND a[1] AND a[2])) = log(p(a[0])*p(a[1])*p(a[2])) = log(p(a[0])) + log(p(a[1])) + log(p(a[2])) ]
            action_log_prob = normal_dist.log_prob(action).sum()

            return action, action_log_prob


    def store_outcome(self, state, next_state, action_log_prob, reward, done):
        """
        Save the step of the trajectory in the class attributes. Store all the trajectory steps together
        """
        self.states.append(torch.from_numpy(state).float())
        self.next_states.append(torch.from_numpy(next_state).float())
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(torch.Tensor([reward]))
        self.done.append(done)


import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal


def discount_rewards(r, gamma): #r=reward ricevute nell'episodio da scontare, gamma= fattore di sconto (se è circa 0 da impo alle reward immediate, 1 future)
    discounted_r = torch.zeros_like(r) #inizializzo con stessa forma di r (tensore)
    running_add = 0 #accumula i ritorni scontati
    #Ciclo Inverso: La funzione calcola il ritorno scontato partendo dalla fine della sequenza di ricompense (dal tempo finale) verso l'inizio.
    #Questo è un passaggio critico per garantire che ogni ritorno scontato dipenda dalle ricompense future, che è l'essenza del reinforcement learning.
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
            Actor network: prende come input lo stato e produce una distribuzione normale dalle quali campionare l'azione
            ha tre fully connected layers e uno stato di attivazione
        """
        self.fc1_actor = torch.nn.Linear(state_space, self.hidden)
        self.fc2_actor = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_actor_mean = torch.nn.Linear(self.hidden, action_space) #restituisce la media dell'azione da prendere, che rappresenta il centro della distribuzione normale
        
        # Learned standard deviation for exploration at training time 
        self.sigma_activation = F.softplus #per avere sigma>=0
        init_sigma = 0.5
        self.sigma = torch.nn.Parameter(torch.zeros(self.action_space)+init_sigma) #non è un layer. La deviazione standard la aggiorno insieme ai pesi del neural network.
                                                                                   #mentre per la media aggiorno i pesi della rete. Voglio sigma indipendente dallo stato, mentre la media dipende dalo stato. Sigma è parametro di esplorazione, non dipende da dove sei                                       

        
        """Critic network: il critico cerca di stimare il valore (o la "votazione") dello stato dato.
        Questo valore viene usato per calcolare l'advantage (vantaggio) che è la differenza tra il ritorno scontato e il valore predetto dallo stato.
        Critic network"""
         # TASK 3: critic network for actor-critic algorithm
        self.fc1_critic = torch.nn.Linear(state_space, self.hidden)
        self.fc2_critic = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_critic_value = torch.nn.Linear(self.hidden, 1)

        
        self.init_weights()


    # inizializza i pesi della rete neurale utilizzando una distribuzione normale per i pesi e zeri per i bias
    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)


    def forward(self, x): #x è lo stato in cui si trova l'agente
        """
            Actor
        """
        x_actor = self.tanh(self.fc1_actor(x)) #primo layer. Output di dim (64,), tanh trasforma tra -1 e 1 per stabilizzare
        x_actor = self.tanh(self.fc2_actor(x_actor))
        action_mean = self.fc3_actor_mean(x_actor)

        # La distribuzione normale è utilizzata per la stocasticità nelle azioni (in modo che l'agente esplori l'ambiente).
        sigma = self.sigma_activation(self.sigma)
        normal_dist = Normal(action_mean, sigma)


        """
            Critic
        """
       # TASK 3: forward in the critic network
        x_critic = self.tanh(self.fc1_critic(x))
        x_critic = self.tanh(self.fc2_critic(x_critic))
        state_value = self.fc3_critic_value(x_critic).squeeze(-1)  # Output: scalare

        return normal_dist, state_value #restituisce la distr da cui campionerà l'agente, nel metodo get_action


class Agent(object):
    def __init__(self, policy, device='cpu'):
        self.train_device = device
        self.policy = policy.to(self.train_device) #rete neurale di tipo Policy che l'agente userà per prendere decisioni
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3) #ottimizzatore Adam per aggiornare i pesi della rete neurale

        self.gamma = 0.99
        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.rewards = []
        self.done = []


    def update_policy(self):
        #Questo metodo converte tutte le variabili memorizzate in tensori di PyTorch e le invia al dispositivo di allenamento (CPU o GPU).
        action_log_probs = torch.stack(self.action_log_probs, dim=0).to(self.train_device).squeeze(-1)
        states = torch.stack(self.states, dim=0).to(self.train_device).squeeze(-1)
        next_states = torch.stack(self.next_states, dim=0).to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        done = torch.Tensor(self.done).to(self.train_device)

        #Una volta che i dati sono stati utilizzati per aggiornare la politica, vengono svuotati.
        self.states, self.next_states, self.action_log_probs, self.rewards, self.done = [], [], [], [], []

        #
        # TASK 2:
        if algorithm == 'reinforce':
            returns = discount_rewards(rewards, self.gamma) #trova la G, vettore in cui in ogni posizione c'è la G_t
            #normalize it
            returns -= returns.mean()
            returns/= returns.std() #così l'ho normalizzato

            #compute backpropagation:
            loss_fn =-torch.mean(action_log_probs * returns) #loss della rete neurale per aggiornare i teta 
            self.optimizer.zero_grad() #riazzera i gradienti
            loss_fn.backward()   #Compute the gradients of the loss w.r.t. each parameter
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(),1)
            self.optimizer.step()   #Compute a step of the optimization algorithm. Aggiornamento di theta

            #   - compute discounted returns
            #   - Calcola le ricompense scontate (usando `discount_rewards`)
            #   - compute policy gradient loss function given actions and returns
            #   - compute gradients and step the optimizer
            
       
        #CON BASELINE
        #madda

        if algorithm =='actor-critic':
            #   - compute boostrapped discounted return estimates
            #   - compute advantage terms
            #   - compute actor loss and critic loss
            #   - compute gradients and step the optimizer
            
            # Compute value estimates from critic
            _, state_values = self.policy(states)       #CHIAMO LA FUNZIONE FORWARD DEL NETWORW policy (scorciatoia)
            _, next_state_values = self.policy(next_states)

            # Compute targets using bootstrapped returns:
            # G_t = r_t + γ * V(s_{t+1}) * (1 - done)  if done ==1, no bootstrapping
            done = done.float()
            targets = rewards + self.gamma * next_state_values * (1 - done)

            #detach serve per dire che quella nvariabile non deve essere vista come parametro, così nel calcolo dei gradienti non di tiene in conto
            #tipo: se tu hai usato i parametri per calcolare l'advantage, torch ti calcolerebbe il gradiente anche sue quello. 
            
            # Compute advantage: A_t = G_t - V(s_t)
            advantages = targets.detach() - state_values #potrei normalizzarlo, da vedere se serve

            # Actor loss (Policy Gradient): maximize advantage × log_prob
            actor_loss = -torch.mean(action_log_probs * advantages.detach())

            # Critic loss (Mean Squared Error)
            critic_loss = F.mse_loss(state_values, targets.detach())
            #critic_loss = -torch.mean(state_values* targets.detach())

            # Total loss
            total_loss = actor_loss + critic_loss #nel policy hai prima actor poi critic, qui non è un + effettivo ma una sequenza: 
                                                # quando si fa optimization, quando l'opt si applica ad actor loss, esso va ad aggiornare 
                                                #i parametri che sono stati usati per calcolare quella loss, e ugualmente per il critic. 
                                                #volendo si può anche fare con due ottimizzatori separati 

            # Optimization step
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1) #clippa il gradiente:se ha norma >1
            self.optimizer.step() #pythotch genera tutti i parametri della rete, sia dell'actor che del critic
            
            

        return        


    def get_action(self, state, evaluation=False):
        """ state -> action (3-d), action_log_densities """
        x = torch.from_numpy(state).float().to(self.train_device)

        normal_dist = self.policy(x) #lo stato dell'ambiente viene passato alla rete neurale che con forward restituisce la distribuzione

        if evaluation:  # Return mean succede tipo nel test
            return normal_dist.mean, None

        else:   # Sample from the distribution
            action = normal_dist.sample()

            # Compute Log probability of the action [ log(p(a[0] AND a[1] AND a[2])) = log(p(a[0])*p(a[1])*p(a[2])) = log(p(a[0])) + log(p(a[1])) + log(p(a[2])) ]
            action_log_prob = normal_dist.log_prob(action).sum() #utile per l'algoritmo di policy gradient

            return action, action_log_prob


    def store_outcome(self, state, next_state, action_log_prob, reward, done): #chiamato nel train
        self.states.append(torch.from_numpy(state).float())
        self.next_states.append(torch.from_numpy(next_state).float())
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(torch.Tensor([reward]))
        self.done.append(done)


"""Train an RL agent on the OpenAI Gym Hopper environment using
    REINFORCE and Actor-critic algorithms???
"""
import argparse
import numpy as np
import torch
import gym
import random
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
from env.custom_hopper import *
from agent import Agent, Policy


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--n_episodes', default=5000, type=int, help='Number of training episodes')   ###DEFAULT: 100000
	parser.add_argument('--print_every', default=500, type=int, help='Print info every <> episodes')   ###DEFAULT: 20000
	parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
	parser.add_argument('--plot', default=True, action='store_true', help='Plot the returns')  ###DEFAULT: False
	parser.add_argument('--plot_every', default=75, type=int, help='Plot return every <> episodes')  ###DEFAULT: 500
	parser.add_argument('--trained_model', default=None, type=str, help='Trained policy path')
	parser.add_argument('--threshold', default=700, type=int, help='Return threshold for early model saving')
	parser.add_argument('--baseline', default=0, type=int, help='Value of REINFORCE baseline')  ###DEFAULT: 0
	parser.add_argument('--random_state', default=42, type=int, help='Randomness seed')
	parser.add_argument('--algorithm', default='reinforce', type=str, help='Algorithm to use [reinforce, actor-critic]')
	return parser.parse_args()


def main(args):
	# Seed setting
	random.seed(args.random_state)
	np.random.seed(args.random_state)
	torch.manual_seed(args.random_state)
	torch.cuda.manual_seed_all(args.random_state)

	start_time = time.time()
	env = gym.make('CustomHopper-source-v0') #crea l'ambiente personalizzato e quindi usa le cose definite in custom hopper
	# env = gym.make('CustomHopper-target-v0')
	env.seed(args.random_state)

	print('Action space:', env.action_space)
	print('State space:', env.observation_space)
	print('Dynamics parameters:', env.get_parameters())


	"""
		Training
	"""
	observation_space_dim = env.observation_space.shape[-1] #Estrae le dimensioni degli spazi di osservazione e azione
	action_space_dim = env.action_space.shape[-1]

	#crea l'oggetto policy e oggetto agent che userà nella politica di addestramento
	policy = Policy(observation_space_dim, action_space_dim)
	agent = Agent(policy, device=args.device)

    #
    # TASK 2 and 3: interleave data collection to policy updates
	returns_list_train = []       # lista dei reward per episodio
	mean_returns_train = []       # lista delle medie ogni N episodi

	for episode in range(args.n_episodes):
		done = False
		train_reward = 0
		state = env.reset()  # Reset the environment and observe the initial state
    
		while not done:  # Loop until the episode is over

			#l'agente seleziona un'azione
			action, action_probabilities = agent.get_action(state)
			previous_state = state

			#l'azione viene eseguita nell'ambiente, che restituisce il nuovo stato e la ricompensa 
			state, reward, done, info = env.step(action.detach().cpu().numpy())

			agent.store_outcome(previous_state, state, action_probabilities, reward, done) # qui crea traiettoria
			train_reward += reward #aggiorna punteggio cumulativo della reward

		returns_list_train.append(train_reward)
		agent.update_policy(algorithm=args.algorithm)	
		
		if (episode+1)%args.print_every == 0:
			mean_return = np.mean(returns_list_train[-args.print_every:])
			mean_returns_train.append(mean_return)
			print('Training episode:', episode)
			print('Episode return:', train_reward)


	end_time = time.time()
	print(f"\nTraining completed in {(end_time - start_time) / 60:.2f} minutes ({(end_time - start_time):.2f} seconds).")
	torch.save(agent.policy.state_dict(), "model.mdl") #salvataggio del modello
	np.save("returns.npy", np.array(mean_returns_train))

    # Plot average returns
	plt.figure(figsize=(10,6))
	x = np.arange(args.print_every, args.n_episodes + 1, args.print_every)
	plt.plot(x, mean_returns_train, marker='o', linestyle='-', label='Mean Return')
	plt.xlabel('Episode')
	plt.ylabel('Average Return')
	plt.title(f'Average Training Return over Episodes ({args.algorithm})')
	plt.legend()
	plt.grid(True)
	plt.tight_layout()
	plt.savefig("plots/training_returns_plot.png")
	

if __name__ == '__main__':
	args = parse_args()
	main(args)
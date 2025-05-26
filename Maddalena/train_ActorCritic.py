"""Train an RL agent on the OpenAI Gym Hopper environment using
    REINFORCE and Actor-critic algorithms

	Implement:
	- Threshold che aumenta man mano automaticamete, sulla base delle performance
	- Salvataggio del progresso dell'optimizer per tuning su modello pre-trainato
	- Mettere azioni dipendenti tra loro (?)
"""
import argparse
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import random

import torch
import gym

from tqdm import tqdm
from env.custom_hopper import *
from agent_ActorCritic import Agent, Policy, Value


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
	parser.add_argument('--random_state', default=42, type=int, help='Random seed')  ###DEFAULT: 0

	return parser.parse_args()

args = parse_args()


def plot_returns(numbers_array,return_array, average_array, beginning_array, points, name):
	"""Plot progress of return over episodes"""
	plt.figure(figsize=(12,10))
	plt.title('RETURN')
	plt.plot(numbers_array, return_array, c='lightsteelblue', label=f'Episode return (every {points})')
	plt.plot(numbers_array[1:], average_array[1:], c='red', linestyle='--', label=f'Average over last {points*2} episodes')
	plt.plot(numbers_array, beginning_array, c='lime', linestyle='--', label='Average over episodes from beginning')
	plt.xlabel('Episode')
	plt.ylabel('Return')
	plt.grid()
	plt.legend()
	plt.savefig("./plots/"+name+'_Return',dpi=300)


def main():

	# Seed setting
	random.seed(args.random_state)
	np.random.seed(args.random_state)
	torch.manual_seed(args.random_state)
	torch.cuda.manual_seed_all(args.random_state)

	# Make directory if it does not exist
	Path.mkdir(Path('./models'),exist_ok=True)
	Path.mkdir(Path('./plots'),exist_ok=True)
	#Define model name based on timestamp
	model_name = f'ActorCritic_{args.n_episodes}_b{args.baseline}_'+datetime.now().strftime('%y%m%d_%H-%M-%S')

	#Make environment
	env = gym.make('CustomHopper-source-v0')
	env_target = gym.make('CustomHopper-target-v0')
	env.seed(args.random_state)
	env_target.seed(args.random_state)

	print('Action space:', env.action_space)
	print('State space:', env.observation_space)
	print('Dynamics parameters:', env.get_parameters())

	observation_space_dim = env.observation_space.shape[-1]
	action_space_dim = env.action_space.shape[-1]

	#Create policy and value
	policy = Policy(observation_space_dim, action_space_dim)
	value = Value(observation_space_dim, action_space_dim)

	#Start from a pre-trained policy
	if args.trained_model:
		policy.load_state_dict(torch.load(args.trained_model), strict=True)
	
	#Create agent
	agent = Agent(policy, value, device=args.device, baseline=args.baseline)

	#Initialize data structures for plot data
	numbers = []
	returns = []
	average_returns = []
	average_beginning = []
	every_return = np.zeros((args.plot_every*2))
	avg_returns = 0
	threshold_bool = False

	#Iterate over episodes (print progress bar in terminal)
	for episode in tqdm(range(args.n_episodes)):
		done = False
		train_reward = 0
		state = env.reset()  # Reset the environment and observe the initial state

		#Build trajectory
		while not done:  # Loop until the episode is over
			action, action_probabilities = agent.get_action(state)
			previous_state = state
			state, reward, done, info = env.step(action.detach().cpu().numpy())
			agent.store_outcome(previous_state, state, action_probabilities, reward, done)
			#Update policy
			agent.update_policy()
			train_reward += reward
		
		#Print progress and save partial model every print_every episodes
		if (episode+1)%args.print_every == 0:
			torch.save(agent.policy.state_dict(), "models/"+model_name+'.mdl')
			print('Training episode:', episode+1)
			print('Episode return:', train_reward)

		#Save data for plotting
		every_return[(episode+1)%(args.plot_every*2)-1]=train_reward
		avg_returns = avg_returns*(episode/(episode+1))+train_reward/(episode+1)
		if args.plot and (episode+1)%args.plot_every == 0:
			numbers.append(episode+1)
			returns.append(train_reward)
			average_returns.append(every_return.mean())
			average_beginning.append(avg_returns)

		#Save a partial model if return over threshold
		if not threshold_bool and every_return.mean()>args.threshold:
			print('Threshold reached')
			torch.save(agent.policy.state_dict(), f"models/{model_name}_t{episode}.mdl")
			threshold_bool = True

	#Print the average of the last episodes' returns
	print(f'Average of the last {args.plot_every*2} returns: {average_returns[-1]}')

	#Save final model (overwrite privious savings)
	torch.save(agent.policy.state_dict(), "models/"+model_name+'.mdl')

	#Plot progress if desired
	if args.plot:
		plot_returns(np.array(numbers),np.array(returns),np.array(average_returns),np.array(average_beginning), args.plot_every, model_name)

	

if __name__ == '__main__':
	main()
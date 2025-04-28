"""Train an RL agent on the OpenAI Gym Hopper environment using
    REINFORCE and Actor-critic algorithms
"""
import argparse
import matplotlib.pyplot as plt
from datetime import datetime

import torch
import gym

from tqdm import tqdm
from env.custom_hopper import *
from agent import Agent, Policy


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--n_episodes', default=10000, type=int, help='Number of training episodes')   ###DEFAULT: 100000
	parser.add_argument('--print_every', default=2000, type=int, help='Print info every <> episodes')   ###DEFAULT: 20000
	parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
	parser.add_argument('--plot', default=True, action='store_true', help='Plot the returns')  ###DEFAULT: False
	parser.add_argument('--plot_every', default=50, type=int, help='Plot return every <> episodes')  ###DEFAULT: 500

	return parser.parse_args()

args = parse_args()


def plot_returns(numbers_array,return_array, average_array, beginning_array, points, name):
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

	model_name = datetime.now().strftime('%y%m%d_%H-%M-%S')

	env = gym.make('CustomHopper-source-v0')
	# env = gym.make('CustomHopper-target-v0')

	print('Action space:', env.action_space)
	print('State space:', env.observation_space)
	print('Dynamics parameters:', env.get_parameters())

	"""
		Training
	"""
	observation_space_dim = env.observation_space.shape[-1]
	action_space_dim = env.action_space.shape[-1]

	policy = Policy(observation_space_dim, action_space_dim)
	agent = Agent(policy, device=args.device)

    #
    # TASK 2 and 3: interleave data collection to policy updates
    #

	numbers = []
	returns = []
	average_returns = []
	average_beginning = []
	every_return = np.zeros((args.plot_every*2))
	sum_returns = 0

	for episode in tqdm(range(args.n_episodes)):
		done = False
		train_reward = 0
		state = env.reset()  # Reset the environment and observe the initial state

		while not done:  # Loop until the episode is over

			action, action_probabilities = agent.get_action(state)
			previous_state = state

			state, reward, done, info = env.step(action.detach().cpu().numpy())

			agent.store_outcome(previous_state, state, action_probabilities, reward, done)

			train_reward += reward
		
		if (episode+1)%args.print_every == 0:
			torch.save(agent.policy.state_dict(), "models/"+model_name+'.mdl')
			print('Training episode:', episode+1)
			print('Episode return:', train_reward)

		every_return[(episode+1)%(args.plot_every*2)-1]=train_reward
		sum_returns += train_reward
		if args.plot and (episode+1)%args.plot_every == 0:
			numbers.append(episode+1)
			returns.append(train_reward)
			average_returns.append(every_return.mean())
			average_beginning.append(sum_returns/(episode+1))

		agent.update_policy()

	torch.save(agent.policy.state_dict(), "models/"+model_name+'.mdl')

	if args.plot:
		plot_returns(np.array(numbers),np.array(returns),np.array(average_returns),np.array(average_beginning), args.plot_every, model_name)

	

if __name__ == '__main__':
	main()
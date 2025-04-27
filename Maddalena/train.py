"""Train an RL agent on the OpenAI Gym Hopper environment using
    REINFORCE and Actor-critic algorithms
"""
import argparse
import matplotlib.pyplot as plt

import torch
import gym

from env.custom_hopper import *
from agent import Agent, Policy


def plot_returns(numbers_array,return_array):
	plt.figure(figsize=(12,10))
	plt.title('RETURN')
	plt.plot(numbers_array, return_array)
	plt.xlabel('Episode')
	plt.ylabel('Return')
	plt.grid()
	plt.savefig('Return',dpi=300)

	plt.figure(figsize=(12,10))
	plt.title('CUMULATIVE RETURN')
	plt.plot(numbers_array, np.cumsum(return_array))
	plt.xlabel('Episode')
	plt.ylabel('Cumulative return')
	plt.grid()
	plt.savefig('CumulativeReturn',dpi=300)


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--n_episodes', default=100000, type=int, help='Number of training episodes')   ###############Changed from 100000
	parser.add_argument('--print_every', default=20000, type=int, help='Print info every <> episodes')   ###############Changed from 20000
	parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
	parser.add_argument('--plot', default=False, action='store_true', help='Plot the returns')
	parser.add_argument('--plot_every', default=50, type=int, help='Plot return every <> episodes')

	return parser.parse_args()

args = parse_args()



def main():
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

	for episode in range(args.n_episodes):
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
			print('Training episode:', episode+1)
			print('Episode return:', train_reward)

		if args.plot and (episode+1)%args.plot_every == 0:
			numbers.append(episode+1)
			returns.append(train_reward)


		agent.update_policy()  #Modified

	torch.save(agent.policy.state_dict(), "model2.mdl")

	if args.plot:
		plot_returns(np.array(numbers),np.array(returns))

	

if __name__ == '__main__':
	main()
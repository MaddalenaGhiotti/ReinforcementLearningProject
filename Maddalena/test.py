"""Test an RL agent on the OpenAI Gym Hopper environment"""
import argparse
import random

import torch
import gym

from env.custom_hopper import *
from agent import Agent, Policy

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', required=True, type=str, help='Model path')
	parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
	parser.add_argument('--render', default=True, action='store_true', help='Render the simulator') ###CHANGE DEFAULT TO FALSE
	parser.add_argument('--episodes', default=10, type=int, help='Number of test episodes')
	parser.add_argument('--random_state', default=42, type=int, help='Randomness seed')
	
	return parser.parse_args()

args = parse_args()


def main():

	# Seed setting
	random.seed(args.random_state)
	np.random.seed(args.random_state)
	torch.manual_seed(args.random_state)
	torch.cuda.manual_seed_all(args.random_state)

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

	#Creat and load policy
	policy = Policy(observation_space_dim, action_space_dim)
	policy.load_state_dict(torch.load(args.model), strict=True)

	#Create agent
	agent = Agent(policy, device=args.device)

	#Iterate over episodes
	for episode in range(args.episodes):
		done = False
		test_reward = 0
		state = env.reset()

		#Build trajectory
		while not done:
			action, _ = agent.get_action(state, evaluation=True)
			state, reward, done, info = env.step(action.detach().cpu().numpy())
			if args.render:  
				env.render()  #Show rendering
			test_reward += reward

		#Print results
		print(f"Episode: {episode} | Return: {test_reward}")
	

if __name__ == '__main__':
	main()
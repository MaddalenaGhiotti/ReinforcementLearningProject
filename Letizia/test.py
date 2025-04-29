"""Test an RL agent on the OpenAI Gym Hopper environment"""
import argparse
import torch
import gym
import numpy as np
import matplotlib.pyplot as plt

from env.custom_hopper import *
from Letizia.agent import Agent, Policy

def parse_args():  # parsing vuol dire che stiamo prendendo gli argomenti da terminale
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=None, type=str, help='Model path')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
    parser.add_argument('--render', default=False, action='store_true', help='Render the simulator')
    parser.add_argument('--episodes', default=10, type=int, help='Number of test episodes')

    return parser.parse_args()

args = parse_args()

def main():
    env = gym.make('CustomHopper-source-v0')
    # env = gym.make('CustomHopper-target-v0')

    print('Action space:', env.action_space)
    print('State space:', env.observation_space)
    print('Dynamics parameters:', env.get_parameters())

    observation_space_dim = env.observation_space.shape[-1]
    action_space_dim = env.action_space.shape[-1]

    policy = Policy(observation_space_dim, action_space_dim)
    policy.load_state_dict(torch.load(args.model), strict=True)

    agent = Agent(policy, device=args.device)

    test_returns = []

    for episode in range(args.episodes):
        done = False
        test_reward = 0
        state = env.reset()

        while not done:
            action, _ = agent.get_action(state, evaluation=True)
            state, reward, done, info = env.step(action.detach().cpu().numpy())

            if args.render:
                env.render()

            test_reward += reward

        test_returns.append(test_reward)
        print(f"Episode: {episode} | Return: {test_reward}")

    # Plot dei returns del test
    plt.figure(figsize=(10, 6))
    plt.plot(range(args.episodes), test_returns, marker='o', linestyle='-', color='green', label='Test Episode Return')
    plt.xlabel('Test Episode')
    plt.ylabel('Return')
    plt.title('Test Return over Episodes (Actor-Critic)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("/mnt/c/Users/letig/Desktop/MachineLearning/progetto/ReinforcementLearningProject/test_returns_plot.png")  
    

if __name__ == '__main__':
    main()

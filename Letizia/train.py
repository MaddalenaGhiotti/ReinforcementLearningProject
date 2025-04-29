"""Train an RL agent on the OpenAI Gym Hopper environment using
    REINFORCE and Actor-critic algorithms
"""
import argparse
import numpy as np
import torch
import gym
import matplotlib.pyplot as plt
import time


from env.custom_hopper import *
from Letizia.agent import Agent, Policy

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-episodes', default=100000, type=int, help='Number of training episodes')
    parser.add_argument('--print-every', default=20000, type=int, help='Print info every <> episodes')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')

    return parser.parse_args()

args = parse_args()
returns_list_train = []

def main():
    env = gym.make('CustomHopper-source-v0')
    # env = gym.make('CustomHopper-target-v0')
    start_time = time.time()


    print('Action space:', env.action_space)
    print('State space:', env.observation_space)
    print('Dynamics parameters:', env.get_parameters())

    observation_space_dim = env.observation_space.shape[-1]
    action_space_dim = env.action_space.shape[-1]

    policy = Policy(observation_space_dim, action_space_dim)
    agent = Agent(policy, device=args.device)

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

        returns_list_train.append(train_reward)
        agent.update_policy()

        if (episode + 1) % args.print_every == 0:
            print('Training episode:', episode)
            print('Episode return:', train_reward)
    end_time = time.time()
    print(f"\nTraining completed in {(end_time-start_time) / 60:.2f} minutes ({(end_time-start_time):.2f} seconds).")	


    # save model and returns
    torch.save(agent.policy.state_dict(), "model.mdl")
    np.save("returns.npy", np.array(returns_list_train))

    # Plot returns
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, args.n_episodes + 1), returns_list_train, marker='o', linestyle='-', color='blue', label='Training Return')
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title('Training Return over Episodes (Actor-Critic)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("/mnt/c/Users/letig/Desktop/MachineLearning/progetto/ReinforcementLearningProject/training_returns_plot.png")
  


if __name__ == '__main__':
    main()

"""Train an RL agent on the OpenAI Gym Hopper environment using
    REINFORCE and Actor-critic algorithms
"""
import argparse
import numpy as np
import torch
import gym
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

from env.custom_hopper import *
from Letizia.agent import Agent, Policy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-episodes', default=100000, type=int, help='Number of training episodes')
    parser.add_argument('--print-every', default=20000, type=int, help='Print info every <> episodes')
    parser.add_argument('--device', default='cpu', type=str, help='Network device [cpu, cuda]')
    parser.add_argument('--algorithm', default='reinforce', type=str, help='Algorithm to use [reinforce, actor-critic]')
    return parser.parse_args()


def main(args):
    env = gym.make('CustomHopper-source-v0', train_mode=True)
    # env = gym.make('CustomHopper-target-v0', train_mode=True)
    start_time = time.time()

    print('Action space:', env.action_space)
    print('State space:', env.observation_space)
    print('Dynamics parameters:', env.get_parameters())

    observation_space_dim = env.observation_space.shape[-1]
    action_space_dim = env.action_space.shape[-1]

    policy = Policy(observation_space_dim, action_space_dim)
    agent = Agent(policy, device=args.device)

    returns_list_train = []       # lista dei reward per episodio
    mean_returns_train = []       # lista delle medie ogni N episodi

    for episode in range(args.n_episodes):
        done = False
        train_reward = 0
        state = env.reset()
        
        if args.algorithm == 'actor-critic':
            agent.reset_I()
        

        while not done:
            action, action_probabilities = agent.get_action(state)
            previous_state = state

            state, reward, done, info = env.step(action.detach().cpu().numpy())

            agent.store_outcome(previous_state, state, action_probabilities, reward, done)
            if args.algorithm == 'actor-critic':
                agent.update_policy(algorithm=args.algorithm)
            train_reward += reward

        returns_list_train.append(train_reward)
        if args.algorithm == 'reinforce':
            agent.update_policy(algorithm=args.algorithm)

        if (episode + 1) % args.print_every == 0:
            mean_return = np.mean(returns_list_train[-args.print_every:])
            mean_returns_train.append(mean_return)
            print('Training episode:', episode + 1)
            print(f'Mean return (last {args.print_every} episodes): {mean_return:.2f}')

    end_time = time.time()
    print(f"\nTraining completed in {(end_time - start_time) / 60:.2f} minutes ({(end_time - start_time):.2f} seconds).")

    # Save model and returns
    torch.save(agent.policy.state_dict(), "model.mdl")
    np.save("returns.npy", np.array(mean_returns_train))

    # Plot average returns
    plt.figure(figsize=(10, 6))
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

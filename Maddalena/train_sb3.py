"""Sample script for training a control policy on the Hopper environment
   using stable-baselines3 (https://stable-baselines3.readthedocs.io/en/master/)

    Read the stable-baselines3 documentation and implement a training
    pipeline with an RL algorithm of your choice between PPO and SAC.
"""
import argparse
import matplotlib.pyplot as plt
from datetime import datetime

import torch
import gym

from tqdm import tqdm
from env.custom_hopper import *
from stable_baselines3 import PPO

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_episodes', default=10, type=int, help='Number of test episodes')   ###DEFAULT: 100000   
    parser.add_argument('--timesteps', default=100000, type=int, help='Number of training episodes')   ###DEFAULT: 2 000 000   
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
    parser.add_argument('--render', default=True, action='store_true', help='Render the simulator') ###CHANGE DEFAULT TO FALSE

    parser.add_argument('--trained_model', default=None, type=str, help='Trained policy path')

    parser.add_argument('--print_every', default=5000, type=int, help='Print info every <> episodes')   ###DEFAULT: 20000
    parser.add_argument('--plot', default=True, action='store_true', help='Plot the returns')  ###DEFAULT: False
    parser.add_argument('--plot_every', default=75, type=int, help='Plot return every <> episodes')  ###DEFAULT: 500
    parser.add_argument('--threshold', default=700, type=int, help='Return threshold for early model saving')

    return parser.parse_args()

args = parse_args()


def main():    
    #Define model name based on timestamp
    #model_name = f'ActorCritic_{args.n_episodes}_b{args.baseline}_'+datetime.now().strftime('%y%m%d_%H-%M-%S')
    
    train_env = gym.make('CustomHopper-source-v0')
    test_env = gym.make('CustomHopper-target-v0')
    
    print('State space:', train_env.observation_space)  # state-space
    print('Action space:', train_env.action_space)  # action-space
    print('Dynamics parameters:', train_env.get_parameters())  # masses of each link of the Hopper

    #
    # TASK 4 & 5: train and test policies on the Hopper env with stable-baselines3
    #

    model = PPO('MlpPolicy', train_env, verbose=1)
    model.learn(total_timesteps=args.timesteps)

    for episode in range(args.n_episodes):
        done= False
        obs = train_env.reset()
        while not done:
            action, states = model.predict(obs)
            obs, reward, done, info = train_env.step(action)
            if args.render:
                train_env.render()

if __name__ == '__main__':
    main()
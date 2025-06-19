"""Script to train a control policy on the Hopper environment using PPO from Stable Baselines3.
"""
import gym
from env.custom_hopper import *
from stable_baselines3 import PPO
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Train a control policy on the Hopper environment')
    parser.add_argument('--train_env', type=str, default='source', help='Environment ID [source,target]')
    parser.add_argument('--UDR', type=bool, default=False, help='Use UDR (Uniform Domain Randomization [True, False]')
    
    if hasattr(__import__('builtins'), '__IPYTHON__'):
        return parser.parse_args([])
    return parser.parse_args()



def main():

    args = parse_args()
    train_env = args.train_env

    if train_env == 'source':
        # Create the source environment
        train_env = gym.make('CustomHopper-source-v0', train_mode=True, use_udr=args.UDR)
    elif train_env == 'target':
        # Create the target environment
        train_env = gym.make('CustomHopper-target-v0', train_mode=True, use_udr=args.UDR)

    print('State space:', train_env.observation_space)  # state-space
    print('Action space:', train_env.action_space)  # action-space
    print('Dynamics parameters:', train_env.get_parameters())  # masses of each link of the Hopper

    model = PPO('MlpPolicy', 
                train_env,
                verbose=1,
                learning_rate=1e-3,
                n_steps=4096,
                batch_size=256,
                n_epochs=10,
                tensorboard_log=None,
                seed=42)    
    
    # Train the model   
    model.learn(total_timesteps=200_000, tb_log_name=model_path)

    # Save the model
    model_path="ppo_hopper" 
    model.save(model_path)


if __name__ == '__main__':
    main()
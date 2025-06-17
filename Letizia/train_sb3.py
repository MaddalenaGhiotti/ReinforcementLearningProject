"""Sample script for training a control policy on the Hopper environment
   using stable-baselines3 (https://stable-baselines3.readthedocs.io/en/master/)

    Read the stable-baselines3 documentation and implement a training
    pipeline with an RL algorithm of your choice between PPO and SAC.
"""
import gym
from env.custom_hopper import *
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.evaluation import evaluate_policy
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Train a control policy on the Hopper environment')
    parser.add_argument('--train_env', type=str, default='source', help='Environment ID [source,target]')
    parser.add_argument('--algo', type=str, default='PPO', choices=['PPO', 'SAC'], help='RL algorithm to use [PPO, SAC]')
    
    if hasattr(__import__('builtins'), '__IPYTHON__'):
        return parser.parse_args([])
    return parser.parse_args()



def main():

    args = parse_args()
    train_env = args.train_env

    algo = args.algo

    if train_env == 'source':
        # Create the source environment
        train_env = gym.make('CustomHopper-source-v0', train_mode=True)
    elif train_env == 'target':
        # Create the target environment
        train_env = gym.make('CustomHopper-target-v0', train_mode=True)



    print('State space:', train_env.observation_space)  # state-space
    print('Action space:', train_env.action_space)  # action-space
    print('Dynamics parameters:', train_env.get_parameters())  # masses of each link of the Hopper

    #
    # TASK 4 & 5: train and test policies on the Hopper env with stable-baselines3
    #
    if algo == 'PPO':
        # PPO algorithm
        model = PPO('MlpPolicy', 
                    train_env,
                    verbose=1,
                    learning_rate=1e-3,
                    n_steps=4096,
                    batch_size=256,
                    n_epochs=10,
                    tensorboard_log=None,
                    seed=42)    
    elif algo == 'SAC':
        # SAC algorithm
        model = SAC('MlpPolicy', 
                    train_env,
                    verbose=0,
                    learning_rate=1e-3,
                    buffer_size=1000000,
                    batch_size=256,
                    tau=0.005,
                    gamma=0.99,
                    train_freq=(1, 'step'),
                    gradient_steps=1,
                    learning_starts=1000,
                    tensorboard_log="./sac_hopper_tensorboard/",
                    seed=42)
        
    # Train the model   
    # PPO    
    model_path="ppo_hopper" if algo == 'PPO' else "sac_hopper"
    model.learn(total_timesteps=200_000, tb_log_name=model_path)

    # Save the model
    model.save(model_path)


if __name__ == '__main__':
    main()
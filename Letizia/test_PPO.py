"""Evaluate a trained PPO policy on the Hopper environment"""
import gym
from env.custom_hopper import *
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a trained policy on the Hopper environment')
    parser.add_argument('--test_env', type=str, default='target', help='Environment ID [source,target]')
    return parser.parse_args()

def main():
    args = parse_args()

    # test environment
    if args.test_env == 'source':
        test_env = gym.make('CustomHopper-source-v0', train_mode=False)
    elif args.test_env == 'target':
        test_env = gym.make('CustomHopper-target-v0', train_mode=False)

    test_env.seed(42)
    test_env = Monitor(test_env)  

    # load the model
    model_path = 'ppo_hopper' 
    model = PPO.load(model_path)

    # evaluate the model
    mean_reward, std_reward = evaluate_policy(
        model,
        test_env,
        n_eval_episodes=50,
        deterministic=True,
        render=False
    )

    print(f"Mean reward over 50 episodes: {mean_reward} ± {std_reward}")

if __name__ == '__main__':
    main()

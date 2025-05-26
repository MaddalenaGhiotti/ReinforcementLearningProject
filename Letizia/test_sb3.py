
import gym
from env.custom_hopper import *
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.evaluation import evaluate_policy
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a trained policy on the Hopper environment')
    parser.add_argument('--test_env', type=str, default='CustomHopper-target-v0', help='Environment ID [source,target]')
    parser.add_argument('--algo', type=str, default='PPO', choices=['PPO', 'SAC'], help='RL algorithm used in the training [PPO, SAC]')
    return parser.parse_args()

def main():
    args = parse_args()

    # test environment
    if args.test_env == 'source':
        test_env = gym.make('CustomHopper-source-v0', train_mode=False)
    elif args.test_env == 'target':
        test_env = gym.make('CustomHopper-target-v0', train_mode=False)
    else:
        test_env = gym.make(args.test_env, train_mode=False)

    # load the model
    model_path = 'ppo_hopper' if args.algo == 'PPO' else 'sac_hopper'
    model = PPO.load(model_path) if args.algo == 'PPO' else SAC.load(model_path)

    # evaluate the model
    mean_reward, std_reward = evaluate_policy(
        model,
        test_env,
        n_eval_episodes=50,
        deterministic=True,
        render=True
    )

    print(f"Mean reward over 50 episodes: {mean_reward} ± {std_reward}")

if __name__ == '__main__':
    main()

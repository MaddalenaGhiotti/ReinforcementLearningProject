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
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_episodes', default=10, type=int, help='Number of test episodes')   ###DEFAULT: 100000   
    parser.add_argument('--timesteps', default=50000, type=int, help='Number of training episodes')   ###DEFAULT: 2e6  
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
    parser.add_argument('--render', default=True, action='store_true', help='Render the simulator') ###CHANGE DEFAULT TO FALSE
    parser.add_argument('--trained_model', default=None, type=str, help='Trained model path')
    parser.add_argument('--env_train', default='source', type=str, help='Env for training (source or target)')
    parser.add_argument('--env_test', default='source', type=str, help='Env for test (source or target)')
    parser.add_argument('--random_state', default=16, type=int, help='Randomness seed')

    parser.add_argument('--print_every', default=5000, type=int, help='Print info every <> episodes')   ###DEFAULT: 20000
    parser.add_argument('--plot', default=True, action='store_true', help='Plot the returns')  ###DEFAULT: False
    parser.add_argument('--plot_every', default=75, type=int, help='Plot return every <> episodes')  ###DEFAULT: 500
    parser.add_argument('--threshold', default=700, type=int, help='Return threshold for early model saving')

    return parser.parse_args()

args = parse_args()


def main():    
    #Define model name based on timestamp
    model_name = f'PPO_'+datetime.now().strftime('%y%m%d_%H-%M-%S')
    
    train_env = gym.make(f'CustomHopper-{args.env_train}-v0')
    test_env = gym.make(f'CustomHopper-{args.env_test}-v0')
    train_env.seed(args.random_state)
    test_env.seed(args.random_state)
    train_env = Monitor(train_env, filename='PPOresults/'+model_name)
    
    print('State space:', train_env.observation_space)  # state-space
    print('Action space:', train_env.action_space)  # action-space
    print('Dynamics parameters:', train_env.get_parameters())  # masses of each link of the Hopper

    #
    # TASK 4 & 5: train and test policies on the Hopper env with stable-baselines3
    #

    if args.trained_model: #If pre-trained
        #Define the model
        model = PPO.load('models/'+args.trained_model)
    else:
        #Define the model
        model = PPO('MlpPolicy', train_env, verbose=1, seed=args.random_state,
                    learning_rate=1e-3,
                    n_steps=2048,
                    batch_size=64,
                    n_epochs=20) #tensorboard_log="./ppo_hopper_tensorboard/"
    
        #Train the model
        model.learn(total_timesteps=args.timesteps, tb_log_name=model_name)
        #Save the model
        model.save("models/"+model_name)

    
    # Evaluate (test) the policy
    eval_returns, eval_length = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=args.n_episodes,
        deterministic=True,
        return_episode_rewards=True,
        render=True  # True se vuoi visualizzare
    )

    #Print results
    print('\nRETURNS')
    for ep, (ret, len) in enumerate(zip(eval_returns,eval_length)):
        print(f"Episode: {ep} | Return: {ret}, Length: {len}")
    print(f'\nAverage return: {sum(eval_returns)/args.n_episodes} +- {np.std(eval_returns)}')
    



    """
    tot_return = 0
    for episode in range(args.n_episodes):
        done= False
        obs = test_env.reset()
        ep_return = 0
        while not done:
            action, states = model.predict(obs)
            obs, reward, done, info = test_env.step(action)
            if args.render:
                test_env.render()
            ep_return += reward
        print(f'Episode {episode}: {ep_return}')
        tot_return += ep_return
    print(f'\nAverage return: {tot_return/args.n_episodes}')
    """   


if __name__ == '__main__':
    main()
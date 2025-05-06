"""Sample script for training a control policy on the Hopper environment
   using stable-baselines3 (https://stable-baselines3.readthedocs.io/en/master/)

    Read the stable-baselines3 documentation and implement a training
    pipeline with an RL algorithm of your choice between PPO and SAC.
"""
import gym
from env.custom_hopper import *
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.evaluation import evaluate_policy


def main():
    train_env = gym.make('CustomHopper-source-v0')

    print('State space:', train_env.observation_space)  # state-space
    print('Action space:', train_env.action_space)  # action-space
    print('Dynamics parameters:', train_env.get_parameters())  # masses of each link of the Hopper

    #
    # TASK 4 & 5: train and test policies on the Hopper env with stable-baselines3
    #
    model=PPO('MlpPolicy', 
              train_env,
              verbose=1,
              learning_rate=1e-3,
              n_steps=2048,
              batch_size=64,
              n_epochs=20,
              tensorboard_log="./ppo_hopper_tensorboard/")
    
    model.learn(total_timesteps=2e5, tb_log_name="PPO_Hopper")
    model.save("ppo_hopper")
    del model  # delete trained model to demonstrate loading

    # Load the saved model
    model = PPO.load("ppo_hopper")

    # Evaluate the policy
    mean_reward, std_reward = evaluate_policy(
        model,
        train_env,
        n_eval_episodes=50,
        deterministic=True,
        render=True  # True se vuoi visualizzare
    )

    print(f"Mean reward over 50 episodes: {mean_reward} ± {std_reward}")




if __name__ == '__main__':
    main()
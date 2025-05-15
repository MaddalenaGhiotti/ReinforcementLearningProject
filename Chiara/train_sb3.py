"""Sample script for training a control policy on the Hopper environment
   using stable-baselines3 (https://stable-baselines3.readthedocs.io/en/master/)

    Read the stable-baselines3 documentation and implement a training
    pipeline with an RL algorithm of your choice between PPO and SAC.

import gym
from env.custom_hopper import *
from stable_baselines3 import PPO  # o SAC


def main():
    train_env = gym.make('CustomHopper-source-v0')

    print('State space:', train_env.observation_space)  # state-space
    print('Action space:', train_env.action_space)  # action-space
    print('Dynamics parameters:', train_env.get_parameters())  # masses of each link of the Hopper

    #
    # TASK 4 & 5: train and test policies on the Hopper env with stable-baselines3
    #

if __name__ == '__main__':
    main()"""
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from env.custom_hopper import *  # Assicurati che il tuo custom env sia correttamente importabile
from stable_baselines3.common.evaluation import evaluate_policy
import argparse

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
    # 1. Crea l'ambiente di training
    train_env = gym.make('CustomHopper-source-v0')
    train_env = Monitor(train_env, filename = "monitor_train.csv")  # utile per logging e metriche
     
    # 2. Crea l'ambiente di valutazione
    eval_env = gym.make('CustomHopper-source-v0')
    eval_env = Monitor(eval_env, filename = "monitor_eval.csv")

    # 3. Callback per valutazione periodica e salvataggio modello
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./ppo_hopper_best_model',
        log_path='./logs/',
        eval_freq=10000,  # ogni tot step
        deterministic=True,
        render=False
    )

    # 4. Inizializza PPO con iperparametri base
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        tensorboard_log="./ppo_tensorboard/"
    )

    # 5. Allena il modello
    model.learn(total_timesteps=200_000, callback=eval_callback)

    # 6. Salva il modello finale
    model.save("ppo_hopper_final")

    print("Training completato. Modello salvato.")

   
    # Evaluate (test) the policy
    eval_returns, eval_length = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=args.n_episodes,
        deterministic=True,
        return_episode_rewards=True,
        render=True  # True se vuoi visualizzare
    )

if __name__ == '__main__':
    main()

    """"Per caricare i modelli in seguito
    from stable_baselines3 import PPO

# Carica modello finale
model = PPO.load("ppo_hopper_final")

# Carica modello migliore
model = PPO.load("./ppo_hopper_best_model/best_model")
"""

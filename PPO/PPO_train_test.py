import os
import gym


import numpy as np
import torch
import time
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from Letizia.env.custom_hopper import *

# ------------------------------------------------------------------
class PPOTrainer:
    """
    Wrapper to train and evaluate PPO on CustomHopper-{source|target}-v0.
    """

    def __init__(
        self,
        train_domain = "source",
        test_domain = "target",
        seed = 42,
        total_timesteps = 100_000,
        model_path = "ppo_hopper",
        learning_rate=1e-3,
        n_steps=2048,
        batch_size=64,
        n_epochs=20,
        verbose=0,
        n_eval_episodes=50,
        tensorboard_log='PPO_hopper_tensorboard',
        use_udr=False
    ) -> None:

        if train_domain not in {"source", "target"}:
            raise ValueError("train_domain must be 'source' or 'target'")
        if test_domain not in {"source", "target"}:
            raise ValueError("test_domain must be 'source' or 'target'")

        self.train_id = f"CustomHopper-{train_domain}-v0"
        self.test_id  = f"CustomHopper-{test_domain}-v0"
        self.seed = seed
        self.total_timesteps = total_timesteps
        self.model_path = model_path
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.verbose = verbose
        self.n_eval_episodes = n_eval_episodes
        self.tensorboard_log = tensorboard_log
        self.use_udr = use_udr

        # reproducibility
        np.random.seed(seed)
        torch.manual_seed(seed)

        # ------------- make envs -----------------
        self.train_env = gym.make(self.train_id, train_mode=True, use_udr=self.use_udr)
        self.train_env.seed(seed)
        self.train_env.action_space.seed(seed)

        self.eval_env  = Monitor(gym.make(self.test_id, train_mode=False))
        self.eval_env.seed(seed)
        self.eval_env.action_space.seed(seed)

        # ------------- build model ---------------
        self.model = PPO(
            "MlpPolicy",
            self.train_env,
            learning_rate=self.learning_rate,
            n_steps=self.n_steps,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            seed=seed,
            verbose=self.verbose,
            tensorboard_log=self.tensorboard_log
        )

    # ------------------------------------------------------------
    def train(self) -> None:
        """
        Train the PPO agent for the configured number of timesteps
        and save it to disk.
        """
        start = time.time()
        self.model.learn(total_timesteps=self.total_timesteps, tb_log_name=self.model_path)
        train_time = time.time() - start
        self.model.save(self.model_path)
        print(" Training completed and model saved.\n")
        return train_time

    # ------------------------------------------------------------
    def evaluate(self, deterministic: bool = True) -> tuple:
        """
        Evaluate the trained model on the selected test domain.
        Returns (mean_reward, std_reward).
        """
        mean_r, std_r = evaluate_policy(
            self.model, self.eval_env, n_eval_episodes=self.n_eval_episodes, deterministic=deterministic
        )
        print(f"Mean reward on {self.test_id}: {mean_r:.2f} ± {std_r:.2f}")
        return mean_r, std_r

    # ------------------------------------------------------------
    def run(self) -> None:
        """Convenience helper: train then evaluate."""
        train_time = self.train()
        return self.evaluate(), train_time





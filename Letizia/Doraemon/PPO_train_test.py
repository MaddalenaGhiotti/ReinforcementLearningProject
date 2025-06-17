import os
import gym


import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy


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

        # reproducibility
        np.random.seed(seed)
        torch.manual_seed(seed)

        # ------------- make envs -----------------
        self.train_env = gym.make(self.train_id, train_mode=True)
        self.train_env.seed(seed)
        self.train_env.action_space.seed(seed)

        self.eval_env  = Monitor(gym.make(self.test_id, train_mode=False))
        self.eval_env.seed(seed)
        self.eval_env.action_space.seed(seed)

        # ------------- build model ---------------
        self.model = PPO(
            "MlpPolicy",
            self.train_env,
            learning_rate=1e-3,
            n_steps=2048,
            batch_size=64,
            n_epochs=20,
            seed=seed,
            verbose=1,
        )

    # ------------------------------------------------------------
    def train(self) -> None:
        """
        Train the PPO agent for the configured number of timesteps
        and save it to disk.
        """

        self.model.learn(total_timesteps=self.total_timesteps, tb_log_name=self.model_path)
        self.model.save(self.model_path)
        print(" Training completed and model saved.\n")

    # ------------------------------------------------------------
    def evaluate(self, n_episodes: int = 50, deterministic: bool = True) -> tuple:
        """
        Evaluate the trained model on the selected test domain.
        Returns (mean_reward, std_reward).
        """
        mean_r, std_r = evaluate_policy(
            self.model, self.eval_env, n_eval_episodes=n_episodes, deterministic=deterministic
        )
        print(f"Mean reward on {self.test_id}: {mean_r:.2f} ± {std_r:.2f}")
        return mean_r, std_r

    # ------------------------------------------------------------
    def run(self) -> None:
        """Convenience helper: train then evaluate."""
        self.train()
        self.evaluate()





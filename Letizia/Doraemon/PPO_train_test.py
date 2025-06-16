import gym
from env.custom_hopper import *           # assume che registri già gli id
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor


def train_and_test(train_domain="source", test_domain="target", seed=42):
    # ---------- training env ----------
    if train_domain == "source":
        train_env = gym.make("CustomHopper-source-v0", train_mode=True)
    elif train_domain == "target":
        train_env = gym.make("CustomHopper-target-v0", train_mode=True)
    else:
        raise ValueError("train_domain must be 'source' or 'target'")

    train_env.seed(seed); train_env.action_space.seed(seed)

    print("State space :", train_env.observation_space)
    print("Action space:", train_env.action_space)
    print("Default masses:", train_env.get_parameters())

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=1e-3,
        n_steps=2048,
        batch_size=64,
        n_epochs=20,
        seed=seed,
        verbose=1,
    )

    model_path = "ppo_hopper"
    model.learn(total_timesteps=int(1e5), tb_log_name=model_path)
    model.save(model_path)
    print("Training completed and model saved.\n")

    # ---------- evaluation env ----------
    eval_id = "CustomHopper-source-v0" if test_domain == "source" else "CustomHopper-target-v0"
    eval_env = Monitor(gym.make(eval_id, train_mode=False))
    eval_env.seed(seed); eval_env.action_space.seed(seed)

    mean_r, std_r = evaluate_policy(model, eval_env, n_eval_episodes=50, deterministic=True)
    print(f"Mean reward on {test_domain} domain: {mean_r:.2f} ± {std_r:.2f}")


if __name__ == "__main__":
    train_and_test("source", "target")


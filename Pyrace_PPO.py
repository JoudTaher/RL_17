"""
Bonus — PPO Agent for PyRace using Stable-Baselines3

Dependencies (install before running):
    pip install stable-baselines3==2.3.2 shimmy==1.3.0

Run training:
    python Pyrace_PPO.py --mode train

Evaluate saved model:
    python Pyrace_PPO.py --mode eval
"""

import argparse
import os

import gymnasium as gym
import gym_race                          # registers Pyrace-v1 / Pyrace-v3
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

# ─── Config ────────────────────────────────────────────────────────────────────

ENV_ID        = "Pyrace-v3"   # discrete(4) actions, continuous obs, shaped reward
MODEL_DIR     = "models_PPO_v01"
MODEL_NAME    = "ppo_pyrace"
TOTAL_STEPS   = 200_000       # increase for better performance
EVAL_EPISODES = 20

# ─── Reward-shaping wrapper ────────────────────────────────────────────────────

class ShapedRewardWrapper(gym.Wrapper):
    """
    Augments the environment reward with:
      +0.1  * forward distance covered this step
      +5    per checkpoint reached
      -10   on crash
      -0.01 per time-step (encourages efficiency)
    """

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward += 0.1 * info.get("dist", 0)
        reward += 5.0 * info.get("check", 0)
        if info.get("crash", False):
            reward -= 10.0
        reward -= 0.01
        return obs, reward, terminated, truncated, info


# ─── Callback — tracks episode rewards for plotting ───────────────────────────

class RewardLogger(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self._episode_reward = 0.0

    def _on_step(self) -> bool:
        reward = self.locals["rewards"][0]
        done   = self.locals["dones"][0]
        self._episode_reward += float(reward)
        if done:
            self.episode_rewards.append(self._episode_reward)
            self._episode_reward = 0.0
        return True


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_env(shaped: bool = True):
    env = gym.make(ENV_ID)
    if shaped:
        env = ShapedRewardWrapper(env)
    return Monitor(env)


def save_reward_plot(rewards: list, path: str):
    plt.figure(figsize=(10, 4))
    plt.plot(rewards, alpha=0.5, label="episode reward")
    if len(rewards) >= 20:
        rolling = np.convolve(rewards, np.ones(20) / 20, mode="valid")
        plt.plot(range(19, len(rewards)), rolling, label="20-ep rolling mean")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("PPO — Training Reward Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()
    print(f"Reward plot saved → {path}")


# ─── Train ────────────────────────────────────────────────────────────────────

def train(total_steps: int = TOTAL_STEPS):
    os.makedirs(MODEL_DIR, exist_ok=True)
    env = make_env(shaped=True)

    model = PPO(
        policy         = "MlpPolicy",
        env            = env,
        learning_rate  = 3e-4,
        n_steps        = 2048,
        batch_size     = 64,
        n_epochs       = 10,
        gamma          = 0.99,
        gae_lambda     = 0.95,
        clip_range     = 0.2,
        verbose        = 1,
    )

    callback = RewardLogger()
    model.learn(total_timesteps=total_steps, callback=callback)

    save_path = os.path.join(MODEL_DIR, MODEL_NAME)
    model.save(save_path)
    print(f"Model saved → {save_path}.zip")

    if callback.episode_rewards:
        save_reward_plot(
            callback.episode_rewards,
            os.path.join(MODEL_DIR, "ppo_rewards.png"),
        )

    env.close()
    return model, callback.episode_rewards


# ─── Evaluate ─────────────────────────────────────────────────────────────────

def evaluate(episodes: int = EVAL_EPISODES):
    load_path = os.path.join(MODEL_DIR, MODEL_NAME)
    if not os.path.exists(load_path + ".zip"):
        raise FileNotFoundError(
            f"No saved model found at {load_path}.zip — train first."
        )

    env   = make_env(shaped=False)   # raw reward for fair evaluation
    model = PPO.load(load_path, env=env)

    total_rewards, crashes, goals = [], 0, 0

    for ep in range(episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        total_rewards.append(ep_reward)
        crashed = info.get("crash", False)
        crashes += int(crashed)
        goals   += int(not crashed)
        print(f"  ep {ep+1:3d}  reward={ep_reward:7.1f}  crash={crashed}")

    env.close()
    print(
        f"\n--- Evaluation over {episodes} episodes ---\n"
        f"  Mean reward : {np.mean(total_rewards):.2f}\n"
        f"  Max  reward : {np.max(total_rewards):.2f}\n"
        f"  Crashes     : {crashes}\n"
        f"  Goals       : {goals}\n"
    )


# ─── CLI ──────────────────────────────────────────────────────────────────────

def build_parser():
    parser = argparse.ArgumentParser(description="PyRace PPO (Bonus Part 3)")
    parser.add_argument(
        "--mode",
        choices=["train", "eval"],
        default="train",
        help="train a new model or evaluate an existing one",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=TOTAL_STEPS,
        help=f"total environment steps for training (default: {TOTAL_STEPS})",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=EVAL_EPISODES,
        help=f"number of episodes for evaluation (default: {EVAL_EPISODES})",
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()

    if args.mode == "train":
        train(total_steps=args.steps)
    else:
        evaluate(episodes=args.eval_episodes)

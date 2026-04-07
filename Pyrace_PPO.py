"""
Bonus — PPO Agent for PyRace using Stable-Baselines3

Required dependencies:
    pip install stable-baselines3==2.3.2 shimmy==1.3.0

Usage:
    python Pyrace_PPO.py --mode train
    python Pyrace_PPO.py --mode eval
"""

import argparse
import os

import gymnasium as gym
import gym_race  # registers Pyrace-v1 and Pyrace-v3
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

# ─── Config ───────────────────────────────────────────────────────────────────

# Pyrace-v3: continuous obs (5-D normalized radar), discrete(4) actions,
# shaped reward built in (progress + checkpoints + crash penalty + time penalty).
# The AdvancedRewardWrapper below adds further PPO-specific shaping on top.
ENV_ID        = "Pyrace-v3"
MODEL_DIR     = "models_PPO_v01"
MODEL_NAME    = "ppo_pyrace"
TOTAL_STEPS   = 200_000
EVAL_EPISODES = 20
DEFAULT_SEED  = 42

# ─── Advanced Reward Wrapper ──────────────────────────────────────────────────

class AdvancedRewardWrapper(gym.Wrapper):
    """Extra reward shaping applied on top of Pyrace-v3's built-in shaped reward.
    Used only in the PPO pipeline; the DQN file is left entirely unchanged.

    Additions:
    - Wall-proximity penalty: each radar reading below ``wall_threshold``
      (normalised 0-1) incurs a proportional penalty, discouraging tight
      hugging of walls and encouraging safer driving lines.
    - Survival micro-bonus: a small positive signal every step the car stays
      alive, reinforcing longer, crash-free episodes.
    """

    def __init__(self, env, wall_threshold: float = 0.15,
                 wall_penalty: float = 0.5, survival_bonus: float = 0.02):
        super().__init__(env)
        self.wall_threshold = wall_threshold
        self.wall_penalty   = wall_penalty
        self.survival_bonus = survival_bonus

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Wall-proximity penalty: radar observations are normalised to [0, 1];
        # values close to 0 mean the car is near a wall.
        min_radar = float(np.min(obs))
        if min_radar < self.wall_threshold:
            proximity_factor = (self.wall_threshold - min_radar) / self.wall_threshold
            reward -= self.wall_penalty * proximity_factor

        # Survival micro-bonus for every step without a crash.
        if not info.get("crash", False):
            reward += self.survival_bonus

        return obs, reward, terminated, truncated, info


def make_env(seed: int = DEFAULT_SEED) -> gym.Env:
    """Create and wrap the environment with Monitor and AdvancedRewardWrapper."""
    env = gym.make(ENV_ID)
    env = AdvancedRewardWrapper(env)
    env = Monitor(env)
    return env


# ─── Train ────────────────────────────────────────────────────────────────────

def train(total_steps: int = TOTAL_STEPS, seed: int = DEFAULT_SEED):
    os.makedirs(MODEL_DIR, exist_ok=True)
    env = make_env(seed=seed)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=seed,
        # Tuned hyperparameters for Pyrace-v3 with dense reward signal.
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        learning_rate=3e-4,
        clip_range=0.2,
        ent_coef=0.01,   # mild entropy bonus to keep exploration alive
    )
    model.learn(total_timesteps=total_steps)

    save_path = os.path.join(MODEL_DIR, MODEL_NAME)
    model.save(save_path)
    print(f"Model saved → {save_path}.zip")
    env.close()


# ─── Evaluate ─────────────────────────────────────────────────────────────────

def evaluate(episodes: int = EVAL_EPISODES, seed: int = DEFAULT_SEED):
    load_path = os.path.join(MODEL_DIR, MODEL_NAME)
    if not os.path.exists(load_path + ".zip"):
        raise FileNotFoundError(f"No saved model at {load_path}.zip — run train first.")

    env   = make_env(seed=seed)
    model = PPO.load(load_path, env=env)

    rewards, crashes, goals, non_crash_episodes = [], 0, 0, 0
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        ep_reward, done = 0.0, False
        terminated = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        rewards.append(ep_reward)
        crashed = info.get("crash", False)
        crashes += int(crashed)
        # A true "goal" is a lap completed without a crash (terminated cleanly).
        # Non-crash episodes include both goals and timeouts (truncated).
        goal_reached = terminated and not crashed
        goals += int(goal_reached)
        non_crash_episodes += int(not crashed)
        print(
            f"  ep {ep+1:3d}  reward={ep_reward:7.1f}"
            f"  crash={crashed}  goal={goal_reached}"
        )

    env.close()
    print(
        f"\n--- Evaluation ({episodes} episodes) ---\n"
        f"  Mean reward        : {np.mean(rewards):.2f}\n"
        f"  Max  reward        : {np.max(rewards):.2f}\n"
        f"  Crashes            : {crashes}\n"
        f"  Goals (lap done)   : {goals}\n"
        f"  Non-crash episodes : {non_crash_episodes}\n"
    )


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyRace PPO — Bonus Part 3")
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--steps", type=int, default=TOTAL_STEPS)
    parser.add_argument("--eval-episodes", type=int, default=EVAL_EPISODES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    if args.mode == "train":
        train(total_steps=args.steps, seed=args.seed)
    else:
        evaluate(episodes=args.eval_episodes, seed=args.seed)

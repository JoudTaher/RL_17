# Bonus Report — Advanced Reinforcement Learning (Part 3)

## 1. Reward Function Analysis (Part 2)

### The problem with simple rewards

The original (sparse) reward signal gives only minimal feedback:

```
reward = +1  (alive)
reward = -1  (crashed)
```

This is misaligned with the real goal. An agent can collect +1 rewards indefinitely by spinning in circles — it never learns to actually drive forward or avoid crashes efficiently.

Reinforcement learning is *"you get what you reward"*. A bad signal produces bad behaviour.

### Improved reward design (used in both files)

Both `Pyrace_RL_DQN.py` and `Pyrace_PPO.py` use the **`Pyrace-v3`** environment, which has `reward_mode="shaped"` built in. This provides a dense, aligned reward signal:

| Signal | What it does |
|---|---|
| Progress toward next checkpoint | Rewards moving forward |
| Checkpoint reached | Bonus for each waypoint |
| Crash | Large penalty |
| Per-step time penalty | Encourages finishing quickly |

This directly addresses the original flaws:
- **Spinning in circles** no longer earns reward — forward progress is required
- **Crashing** carries a strong penalty
- **Efficiency** is rewarded — less time spent = better score

---

## 2. Bonus — Proximal Policy Optimization (PPO)

### Why move beyond DQN?

DQN approximates Q-values and picks the best discrete action. It works, but it is limited to discrete action spaces and can be unstable.

**PPO** is a policy-gradient method — it directly learns a *policy* π(a | s) that maps states to action probabilities. Key advantages:

| | DQN | PPO |
|---|---|---|
| Action space | Discrete only | Discrete and Continuous |
| Training stability | Sensitive to hyperparameters | Stable (clipped objective) |
| Implementation | From scratch | Library-ready (Stable-Baselines3) |

PPO uses a **clipped objective** to prevent the policy from changing too drastically in one update, making it reliably stable to train.

### Implementation

`Pyrace_PPO.py` wraps the `Pyrace-v3` environment with an `AdvancedRewardWrapper` that adds two extra shaping signals on top of the built-in shaped reward. This extra layer is applied only in the PPO pipeline — the DQN file is left completely unchanged.

| Extra signal | Effect |
|---|---|
| Wall-proximity penalty | Each radar reading below 0.15 (normalised) adds a proportional penalty, discouraging tight wall-hugging |
| Survival micro-bonus | +0.02 per step alive, reinforcing longer crash-free episodes |

```python
from stable_baselines3 import PPO

env = AdvancedRewardWrapper(gym.make("Pyrace-v3"))
env = Monitor(env)

model = PPO(
    "MlpPolicy", env, verbose=1, seed=42,
    n_steps=2048, batch_size=64, n_epochs=10,
    gamma=0.99, gae_lambda=0.95,
    learning_rate=3e-4, clip_range=0.2, ent_coef=0.01,
)
model.learn(total_timesteps=200_000)
model.save("models_PPO_v01/ppo_pyrace")
```

#### Evaluation metrics

The evaluation loop distinguishes three distinct outcomes per episode:

| Metric | Meaning |
|---|---|
| Crashes | Episode ended with a collision |
| Goals (lap done) | `terminated=True` **and** `crash=False` — the car completed a full lap |
| Non-crash episodes | Episodes that ended without a crash (includes timeouts) |

This avoids the earlier pitfall of counting every non-crash episode as a "goal" when some episodes end due to a timeout rather than lap completion.

---

## 3. How to Run

### Install dependencies

```bash
pip install stable-baselines3==2.3.2 shimmy==1.3.0
```

> `stable-baselines3 >= 2.0.0` is required for native `gymnasium` support.

### Train

```bash
python Pyrace_PPO.py --mode train --steps 200000 --seed 42
```

### Evaluate

```bash
python Pyrace_PPO.py --mode eval --eval-episodes 20 --seed 42
```

---

## 4. Summary

| Part | What was done |
|---|---|
| Part 1 — DQN | Discrete-action agent trained from scratch with replay buffer and target network |
| Part 2 — Reward | Switched to `Pyrace-v3` shaped reward: forward progress + checkpoints + crash penalty + time penalty |
| Bonus — PPO | Same `Pyrace-v3` environment + `AdvancedRewardWrapper` (wall-proximity penalty, survival bonus); tuned PPO hyperparameters via Stable-Baselines3; reproducible via `--seed` |


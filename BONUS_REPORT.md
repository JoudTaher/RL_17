# Bonus Report — Advanced Reinforcement Learning (Part 3)

## Overview

This document accompanies **`Pyrace_PPO.py`**, the bonus implementation built on top of the existing DQN work. It explains the motivation, design decisions, and key concepts behind the upgrade.

---

## 1. Limitations of DQN (Part 1 Recap)

The DQN agent from Part 1 works by learning a Q-function — a table of "how good is action *a* in state *s*?" — approximated by a neural network.

| Property | DQN |
|---|---|
| Action space | Discrete only |
| Learning target | Q-values |
| Exploration | ε-greedy |
| Sample efficiency | Moderate (replay buffer helps) |

While DQN performs well, it is limited to a **discrete** set of actions (`0, 1, 2, 3`). Real driving involves smooth, continuous control — steering angles and throttle levels — which discrete actions can only approximate coarsely.

---

## 2. Reward Function Analysis (Part 2)

### Original reward signal

The sparse reward used in the baseline provided only minimal feedback:

```
reward = +1  (still alive)
reward = -1  (crashed)
```

**Problem:** The agent can collect +1 rewards indefinitely by driving in circles, never making meaningful progress. The signal does not distinguish between productive and unproductive behaviour.

### Improved reward design

The shaped reward (already integrated into `Pyrace-v3` and the `ShapedRewardWrapper` in `Pyrace_PPO.py`) addresses each failure mode:

| Signal | Code | Purpose |
|---|---|---|
| Forward progress | `+0.1 * info["dist"]` | Encourages moving ahead |
| Checkpoint bonus | `+5 * info["check"]` | Rewards reaching waypoints |
| Crash penalty | `-10 if info["crash"]` | Strongly discourages collisions |
| Time penalty | `-0.01` per step | Encourages efficiency |

**Why this matters:**  
Reinforcement learning is fundamentally *"you get what you reward"*. A well-aligned reward function guides the agent toward the true objective — driving efficiently around the track — rather than exploiting loopholes in a simple signal.

---

## 3. Proximal Policy Optimization (PPO) — Bonus Algorithm

### What is PPO?

PPO is a **policy-gradient** method that directly learns a policy π(a | s) — a mapping from states to action probabilities — instead of learning Q-values.

```
DQN:  Q(s, a)  →  pick argmax action
PPO:  π(s)     →  sample/select action directly
```

PPO improves on vanilla policy gradients by introducing a **clipped objective** that prevents the policy from changing too drastically in a single update, making training stable without requiring a complex trust-region calculation.

### Why PPO over DQN here?

| Feature | DQN | PPO |
|---|---|---|
| Action space | Discrete | Discrete **and** Continuous |
| Training stability | Sensitive to hyperparams | More stable (clip constraint) |
| Implementation | From scratch | Library-ready |
| On-policy / Off-policy | Off-policy | On-policy |

PPO naturally extends to continuous action spaces, making it a stepping stone toward full continuous control (e.g., DDPG/SAC) with minimal code changes.

### Algorithm sketch

```
For each rollout:
  1. Collect N steps using current policy π_old
  2. Compute advantage estimates A(s, a) using GAE
  3. For K epochs, update π by maximising:
       L_CLIP = E[ min(r_t * A_t,  clip(r_t, 1-ε, 1+ε) * A_t) ]
     where r_t = π(a|s) / π_old(a|s)
  4. Repeat
```

The clipping ensures the updated policy never strays too far from `π_old`, providing a stable learning signal.

---

## 4. Implementation Details

### File: `Pyrace_PPO.py`

| Component | Detail |
|---|---|
| Environment | `Pyrace-v3` (discrete 4-action, continuous 5-D obs) |
| Reward wrapper | `ShapedRewardWrapper` — adds dist/check/crash/time signals |
| Algorithm | `stable_baselines3.PPO` with `MlpPolicy` |
| Network | Two-layer MLP (default 64×64, auto-sized to obs/action space) |
| Default training steps | 200 000 (adjustable via `--steps`) |
| Output | `models_PPO_v01/ppo_pyrace.zip` + reward plot |

### Key hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| `learning_rate` | 3e-4 | Standard PPO starting point |
| `n_steps` | 2048 | Rollout length before update |
| `batch_size` | 64 | Mini-batch for gradient update |
| `n_epochs` | 10 | Update passes per rollout |
| `gamma` | 0.99 | Long-horizon discounting |
| `gae_lambda` | 0.95 | Advantage estimation bias/variance trade-off |
| `clip_range` | 0.2 | Policy update constraint |

---

## 5. How to Run

### Install dependencies

```bash
pip install stable-baselines3==2.3.2 shimmy==1.3.0
```

> **Note:** `stable-baselines3 >= 2.0.0` is required for native `gymnasium` support.  
> `shimmy` provides compatibility shims used internally by SB3.

### Train

```bash
python Pyrace_PPO.py --mode train --steps 200000
```

### Evaluate a saved model

```bash
python Pyrace_PPO.py --mode eval --eval-episodes 20
```

---

## 6. Connecting the Parts

| Part | What it contributes |
|---|---|
| Part 1 — DQN | Baseline discrete-action agent; introduces replay buffer, target network, ε-greedy |
| Part 2 — Reward analysis | Aligns reward with true driving objective; improves signal quality for all algorithms |
| Bonus — PPO | Uses improved reward + stable library algorithm; more stable training, extensible to continuous control |

Each part builds on the previous one. The reward improvement from Part 2 directly benefits the PPO agent, demonstrating that algorithm upgrades and reward design work together.

---

## 7. Conclusion

By moving from DQN to PPO we gain:

- **Stability** — clipped updates prevent catastrophic policy changes
- **Flexibility** — same code extends to continuous actions by swapping `Discrete` for `Box`
- **Practicality** — Stable-Baselines3 provides a production-quality implementation, letting us focus on environment design and reward shaping rather than low-level RL mechanics

The combination of a well-shaped reward (Part 2) and a robust policy-gradient algorithm (PPO, Bonus) produces an agent that learns smoother, more goal-directed driving behaviour than the baseline DQN.

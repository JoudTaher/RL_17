# Bonus Report 

# PPO Agent and Reward Improvements

## 1. Part 2: Reward Function Analysis

The original reward signal was not fully aligned with the goal of efficient driving. A simple reward (e.g., survival or minimal penalties) can lead to undesirable behaviour such as moving without meaningful progress.

To address this, a shaped reward was designed to better reflect the task objective. The improved reward includes:
- forward progress rewards,
- checkpoint bonuses,
- crash penalties,
- and a small time penalty.

This provides more informative feedback and encourages efficient, goal-directed driving.

> Note: The reward redesign was analyzed in Part 2 but **implemented only in the bonus PPO setup** via a wrapper. The Part 1 DQN model was left unchanged to preserve a stable baseline.

---

## 2. Bonus: PPO with Stable-Baselines3

For the bonus, a PPO agent was implemented using Stable-Baselines3.

PPO is a policy-based method that directly learns a mapping from states to actions. Compared to DQN, it is more stable and easier to implement using existing frameworks.

The PPO agent was trained on the shaped-reward PyRace environment.

```python
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
import gym_race

env = Monitor(gym.make("Pyrace-v3"))
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=200_000)
model.save("models_PPO_v01/ppo_pyrace")


## 3. Evaluation Results

The trained PPO agent was evaluated over 20 deterministic episodes:

- **Mean reward:** 4782.65  
- **Max reward:** 4782.65  
- **Crashes:** 0  
- **Successful lap completions:** 20/20  

All episodes produced identical rewards, indicating stable and deterministic behaviour.

The PPO bonus agent was trained using Stable-Baselines3 on the shaped-reward PyRace environment. During evaluation over 20 deterministic episodes, the agent achieved a mean reward of 4782.65, with 0 crashes and 20/20 successful lap completions. The identical reward across episodes indicates stable deterministic behavior from the learned policy.

---

## 4. Reflection

This work highlights two key insights:

- **Reward design is critical:** The agent learns what is incentivized. Poorly aligned rewards lead to ineffective behaviour, regardless of the algorithm used.  
- **Modern RL frameworks are effective:** Using PPO through Stable-Baselines3 simplified implementation while achieving strong and stable performance.  

Overall, combining improved reward design with a robust algorithm resulted in a reliable and effective racing agent.

---

## 5. How to Run the Bonus

### Install requirement.txt

```bash
pip install stable-baselines3==2.3.2 shimmy==1.3.0
python Pyrace_PPO.py --mode train  #to train
python Pyrace_PPO.py --mode eval   # to evaluate


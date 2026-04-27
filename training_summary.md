# Soccer Twos RL Training Summary

This document summarizes the reinforcement learning training pipeline, environment modifications, and curriculum learning strategy we implemented for the Soccer Twos environment.

## 1. Algorithm and Framework

### Proximal Policy Optimization (PPO)
We selected **Proximal Policy Optimization (PPO)** as our core reinforcement learning algorithm. PPO is a state-of-the-art policy gradient method that strikes an excellent balance between ease of tuning, sample complexity, and robustness. 

**Theoretical Background:**
PPO belongs to the family of actor-critic methods. It operates by maintaining two neural networks:
- **Actor Network:** Learns the policy, outputting the best action to take given the current observation of the field.
- **Critic Network:** Learns the value function, estimating the expected total reward from the current state. This helps to calculate "Advantages" (how much better an action was than expected) which reduces the variance of the mathematical updates.

PPO's key mathematical innovation is its **clipped surrogate objective function**. In traditional policy gradient methods, updating the neural network weights too drastically based on a single batch of data can cause the agent to "forget" how to play and collapse the learning process. PPO prevents this by comparing the new policy to the old policy (calculating a probability ratio) and **clipping** this ratio to a small interval. This enforces a "trust region," ensuring that the policy does not change too wildly in a single update step, leading to highly stable, monotonic improvement during our long 12-hour training runs.

### Framework: Ray RLlib (PyTorch)
We implemented PPO using **Ray RLlib**, an industry-standard, highly scalable reinforcement learning library utilizing PyTorch as the backend. RLlib was specifically chosen because it provides native, first-class support for **Multi-Agent Reinforcement Learning (MARL)**. It allowed us to easily map multiple players to a shared policy and distribute the experience collection (rollouts) across 8 concurrent CPU workers on the PACE cluster.

## 2. Environment Modifications: Reward & Observation Spaces

The base Soccer Twos environment suffers from **sparse rewards**, only providing a reward when a goal is scored. To accelerate learning, we implemented a robust **Reward Shaping** wrapper (`RewardShapedWrapper`).

### Reward Shaping Components
We introduced dense, incremental rewards at every step to guide the agent towards scoring. The specific modifications and their underlying hypotheses were:

- **Ball Distance:** 
  - *Modification:* Calculated the Euclidean distance between the agent and the ball. A positive reward was issued if the distance decreased compared to the previous step, and a negative penalty if it increased.
  - *Hypothesis:* The sparse environment only rewards scoring. If the agent doesn't know to approach the ball, it will never score. By providing a breadcrumb trail of rewards for closing the distance to the ball, the agent rapidly learns the basic navigational physics of the field.

- **Possession:**
  - *Modification:* Granted a small, constant positive reward (+0.01) whenever the agent's collision box was in direct contact with the ball.
  - *Hypothesis:* We hypothesized that even if the agent reaches the ball, it might just bounce off it. A possession reward encourages the agent to stick to the ball and learn how to control it and dribble, rather than just colliding randomly.

- **Goal Velocity:**
  - *Modification:* Calculated the velocity vector of the ball and projected it onto the vector pointing directly at the opponent's goal. We rewarded the agent proportionally to the magnitude of this forward velocity.
  - *Hypothesis:* Just holding the ball isn't enough; it must move towards the opponent's goal. By rewarding the exact speed at which the ball approaches the target, we encourage the agent to strike the ball with force in the correct direction, rather than nudging it aimlessly.

- **Goal Proximity:**
  - *Modification:* Similar to ball distance, but measured the distance from the *ball* to the *opponent's goal center*. Rewards were given when the ball moved closer to the goal.
  - *Hypothesis:* Goal velocity only rewards the exact moment of a strike. Goal proximity provides a persistent, state-based reward that tells the agent "the current state of the board is favorable." This encourages the team to keep the ball in the opponent's half of the field.

- **Alignment:**
  - *Modification:* Calculated the angle between the agent's facing direction, the ball, and the opponent's goal. A reward was given when the agent positioned its body directly behind the ball facing the goal.
  - *Hypothesis:* We noticed agents often hit the ball from the wrong angle, knocking it sideways. By explicitly rewarding correct geometric alignment, we aimed to teach the agent the mechanical prerequisite for a clean, powerful shot on goal.

### Observation & Action Space Adjustments
- We utilized `flatten_branched=True` to flatten the Unity MultiDiscrete action space `[3, 3, 3]` into a single `Discrete(27)` action space.
- The observation space was kept at a flat `Box(336,)` shape.
- We encountered Unity port collision issues when initializing dummy environments on Ray workers to read spaces. To fix this, we **hardcoded the PyTorch observation and action spaces** natively in the training scripts.

## 3. Multi-Agent Training Architecture

We transitioned from a single-agent architecture to a **Multi-Agent (MA)** architecture using RLlib's `MultiAgentEnv`.

- We created `MultiAgentTeamVsPolicyWrapper` to interface PyTorch Ray with the 4-player Unity environment.
- The environment natively returns an observation dictionary for players 0, 1, 2, and 3.
- We utilized RLlib's `policy_mapping_fn` to **map multiple players to the same shared Neural Network** (`default_policy`). This allowed both agents on the Blue Team to share knowledge and learn cooperative behavior simultaneously.

## 4. Curriculum Learning Pipeline

To ensure the agent learned progressively complex behaviors without getting stuck in local optima, we designed a **3-Phase Curriculum Learning** pipeline. Each phase resumed training from the best checkpoint of the previous phase.

### Phase 1: Training against Random Opponents (`train_ray_ppo_ma_shaped.py`)
- **Opponent:** The Orange team (agents 2 and 3) selected random actions from the action space.
- **Objective:** Teach the Blue team the basic physics of the game, how to run to the ball, and how to push it towards the goal without organized resistance.

### Phase 2: Training against the Baseline (`train_ray_ppo_ma_shaped_vs_baseline.py`)
- **Opponent:** We injected the provided `TeamAgent` (a pre-trained DQN model) as the `opponent_policy`.
- **Engineering Fix:** We had to implement a `LazyBaselinePolicy` adapter to lazily initialize the `TeamAgent`. This bypassed PyTorch serialization (`pickle`) errors caused by Ray trying to send the baseline neural network across worker nodes. We also intercepted the output to ensure it returned a single integer `Discrete(27)` action instead of a list.
- **Objective:** Force the Blue team to learn how to play against a semi-competent opponent that actively tries to defend and score.

### Phase 3: Self-Play (`train_ray_ppo_ma_shaped_selfplay.py`)
- **Opponent:** The agent itself.
- **Architecture:** We reverted the environment to pure 4-player mode and mapped **all 4 agents** (Blue 0, 1 and Orange 2, 3) to the `"default_policy"`.
- **Objective:** The neural network calculates actions for both sides of the field and learns from all experiences symmetrically. The agent constantly plays against an exact clone of its current self. As the Blue team discovers a new offensive strategy, the weights update, and the Orange team instantly learns that strategy—forcing the Blue team to figure out how to defend against it. This drove the emergence of highly complex, competitive mechanics.

## 5. Final Training Hyperparameters

For the final self-play training run on the PACE cluster, we utilized the following configuration and hyperparameters within PyTorch Ray:

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Algorithm** | PPO | Proximal Policy Optimization |
| **Framework** | Torch | PyTorch Backend |
| **Learning Rate** | `5e-5` | Step size for gradient descent updates |
| **Train Batch Size** | `12000` | Number of experiences used for a single PPO update |
| **Rollout Fragment Length** | `500` | Number of steps each worker collects before sending to the driver |
| **Num Workers (CPU Cores)** | `8` | Number of parallel rollout worker processes |
| **Num Envs per Worker** | `3` | Number of parallel Unity instances per worker |
| **Total Parallel Environments** | `24` | Total environments simulating concurrently |
| **Num GPUs** | `1` | Number of GPUs used for learning updates (NVIDIA V100) |
| **Network Architecture** | `[512]` | Single hidden layer with 512 nodes |
| **Value Function Layer Sharing** | `True` | The Actor and Critic networks share the same hidden layer |

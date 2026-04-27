# Soccer Twos RL Training Summary

This document summarizes the reinforcement learning training pipeline, environment modifications, and curriculum learning strategy we implemented for the Soccer Twos environment.

## 1. Environment Modifications: Reward & Observation Spaces

The base Soccer Twos environment suffers from **sparse rewards**, only providing a reward when a goal is scored. To accelerate learning, we implemented a robust **Reward Shaping** wrapper (`RewardShapedWrapper`).

### Reward Shaping Components
We introduced dense, incremental rewards at every step to guide the agent towards scoring:
- **Ball Distance:** Rewarded agents for moving closer to the ball.
- **Possession:** Granted small rewards when an agent was touching the ball.
- **Goal Velocity:** Rewarded the team when the ball was moving rapidly towards the opponent's goal.
- **Goal Proximity:** Rewarded the team based on how close the ball was to the opponent's goal.
- **Alignment:** Rewarded the agent for positioning itself correctly between the ball and the opponent's goal.

### Observation & Action Space Adjustments
- We utilized `flatten_branched=True` to flatten the Unity MultiDiscrete action space `[3, 3, 3]` into a single `Discrete(27)` action space.
- The observation space was kept at a flat `Box(336,)` shape.
- We encountered Unity port collision issues when initializing dummy environments on Ray workers to read spaces. To fix this, we **hardcoded the PyTorch observation and action spaces** natively in the training scripts.

## 2. Multi-Agent Training Architecture

We transitioned from a single-agent architecture to a **Multi-Agent (MA)** architecture using RLlib's `MultiAgentEnv`.

- We created `MultiAgentTeamVsPolicyWrapper` to interface PyTorch Ray with the 4-player Unity environment.
- The environment natively returns an observation dictionary for players 0, 1, 2, and 3.
- We utilized RLlib's `policy_mapping_fn` to **map multiple players to the same shared Neural Network** (`default_policy`). This allowed both agents on the Blue Team to share knowledge and learn cooperative behavior simultaneously.

## 3. Curriculum Learning Pipeline

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

# CS8803 DRL Team 3 agent

**Authors:** Yiming Chen and Wuyang Du
Example training/testing scripts for the Soccer-Twos environment. This starter code is modified from the example code provided in https://github.com/bryanoliveira/soccer-twos-starter.

Environment-level specification code can be found at https://github.com/bryanoliveira/soccer-twos-env, which may also be useful to reference.

## Requirements

- Python 3.8
- See [requirements.txt](requirements.txt)

## Usage

*Note: For further environment setup details, you may refer to the [original repository](https://github.com/bryanoliveira/soccer-twos-starter).*

### Download Checkpoints
Download the pre-trained checkpoints from [Google Drive](https://drive.google.com/drive/folders/1c8E2l-nQ-mk4OwQ5D6MeKBWwcEVO-Cin?usp=sharing) and place them into a `checkpoints` folder under the root directory.

## High-Level Approach

Our solution utilizes **Proximal Policy Optimization (PPO)** implemented via Ray RLlib with a Multi-Agent architecture. To overcome the sparse rewards inherent in the environment, we applied extensive **Reward Shaping**, granting dense rewards for closing distance to the ball, maintaining possession, and driving the ball towards the opponent's goal with correct geometric alignment.

To ensure stable learning, we designed a **3-Phase Curriculum Learning** pipeline:
1. **Phase 1 (Random):** Training against random opponents to learn basic game physics.
2. **Phase 2 (Baseline):** Training against the provided baseline agent to learn defensive and offensive positioning.
3. **Phase 3 (Self-Play):** Symmetric self-play where the agents play against an exact clone of their current policy, driving the continuous emergence of complex, highly competitive mechanics.

For a full technical breakdown of the architecture, observation modifications, and hyperparameters, please refer to [`training_summary.md`](training_summary.md).

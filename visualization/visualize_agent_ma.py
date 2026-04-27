import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import random
import ray
import ray.rllib.agents.ppo as ppo
from ray.tune.registry import register_env
from soccer_twos import EnvType
from utils import create_rllib_env
import gym
from gym.spaces import Box, Discrete
import numpy as np
import os

def main():
    # Absolute path to the multi-agent checkpoint file
    checkpoint_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "checkpoints", "PPO_SoccerShapedMA_2541d_00000_0_2026-02-21_13-57-19", "checkpoint_001550", "checkpoint-1550"))

    # Register the environment
    register_env("SoccerShapedMA", create_rllib_env)

    # Initialize Ray
    # Force Ray to ignore GPUs to avoid detection errors when nvidia-smi is missing
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    ray.init(ignore_reinit_error=True, num_gpus=0, include_dashboard=False)
    
    obs_space = Box(low=-np.inf, high=np.inf, shape=(336,), dtype=np.float32)
    act_space = Discrete(27)

    # Configuration matches train_ray_ppo_ma_shaped.py but adapted for inference
    config = {
        "env": "SoccerShapedMA",
        "env_config": {
            "flatten_branched": True,
            "shaped_reward": True,
            "variation": EnvType.multiagent_player,
            "team_vs_random_multiagent": True,
            "num_envs_per_worker": 1,
        },
        "multiagent": {
            "policies": {"default_policy": (None, obs_space, act_space, {})},
            "policy_mapping_fn": lambda agent_id, *args, **kwargs: "default_policy",
        },
        "framework": "torch",
        "model": {
            "fcnet_hiddens": [512],
            "vf_share_layers": True,
        },
        "num_gpus": 0,
        "num_workers": 0,  # Local worker for rendering
        "explore": False,
    }

    # Create the Trainer
    print("Creating PPO Trainer...")
    agent = ppo.PPOTrainer(config=config, env="SoccerShapedMA")

    # Restore the checkpoint
    print(f"Restoring checkpoint from {checkpoint_path}...")
    agent.restore(checkpoint_path)

    # Create the environment for visualization
    print("Creating visualization environment...")
    env_config_viz = config["env_config"].copy()
    env_config_viz["worker_id"] = random.randint(200, 500) # Avoid port conflicts if previous run crashed
    env_config_viz["watch"] = True  # Enable watch mode for real-time visualization
    env_config_viz["opponent_policy"] = lambda *_: random.randint(0, 26)
    
    env = create_rllib_env(env_config_viz)

    print(f"Action Space: {env.action_space}")
    print("Starting visualization loop...")
    try:
        obs = env.reset()
        total_reward = {}
        while True:
            # Helper to query the agent for an action
            actions = {}
            if isinstance(obs, dict):
                for agent_id, agent_obs in obs.items():
                    # Agent IDs usually map: 0, 1 -> Blue team; 2, 3 -> Yellow team
                    # But it could be strings like "0", "1". Let's handle both.
                    if str(agent_id) in ["0", "1"]:
                        actions[agent_id] = agent.compute_action(agent_obs)
                    else:
                        # Opponent team plays randomly
                        actions[agent_id] = random.randint(0, 26)
            else:
                # Handle stacked team observation (2 players x 336 dim)
                if len(obs.shape) == 1 and obs.shape[0] == 672:
                    obs_1 = obs[:336]
                    obs_2 = obs[336:]
                    a1 = agent.compute_action(obs_1)
                    a2 = agent.compute_action(obs_2)
                    actions = a1 * 27 + a2
                else:
                    # print(f"Obs shape: {obs.shape}")
                    actions = agent.compute_action(obs)
            
            # Step the environment
            obs, reward, done, info = env.step(actions)

            # Accumulate rewards
            if isinstance(reward, dict):
                for agent_id, r in reward.items():
                    total_reward[agent_id] = total_reward.get(agent_id, 0) + r
            else:
                # Assuming reward is per-agent or total team reward
                # If reward is scalar, just accumulate. If array/list, sum?
                target_rew = sum(reward) if isinstance(reward, (list, tuple)) else reward
                total_reward[0] = total_reward.get(0, 0) + target_rew
            
            is_done = done["__all__"] if isinstance(done, dict) else done
            if is_done:

                print(f"Episode finished. Total Reward: {total_reward}")
                total_reward = {}
                obs = env.reset()
    except KeyboardInterrupt:
        print("\nVisualization stopped by user.")
    finally:
        env.close()
        ray.shutdown()

if __name__ == "__main__":
    main()

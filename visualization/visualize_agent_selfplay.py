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
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import os

class DummyMAEnv(MultiAgentEnv):
    def __init__(self, *args, **kwargs):
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(336,), dtype=np.float32)
        self.action_space = Discrete(27)
    def reset(self):
        return {"0": self.observation_space.sample()}
    def step(self, action_dict):
        obs = {"0": self.observation_space.sample()}
        rew = {"0": 0.0}
        done = {"0": False, "__all__": False}
        info = {"0": {}}
        return obs, rew, done, info

def main():
    # Absolute paths to the multi-agent checkpoint files
    checkpoint_path_blue = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "checkpoints", "PPO_SoccerShapedMASelfPlay_3e80a_00000_0_2026-03-01_17-16-08", "checkpoint_003500", "checkpoint-3500"))
    # checkpoint_path_yellow = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "checkpoints", "PPO_SoccerShapedMABaseline_cfd23_00000_0_2026-02-23_16-51-31", "checkpoint_001650", "checkpoint-1650"))
    checkpoint_path_yellow = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "checkpoints", "PPO_SoccerShapedMASelfPlay_3e80a_00000_0_2026-03-01_17-16-08", "checkpoint_003500", "checkpoint-3500"))

    # Register the environments
    register_env("SoccerShapedMA", create_rllib_env)
    register_env("DummyMAEnv", lambda config: DummyMAEnv())

    # Initialize Ray
    # Force Ray to ignore GPUs to avoid detection errors when nvidia-smi is missing
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    ray.init(ignore_reinit_error=True, num_gpus=0, include_dashboard=False)
    
    obs_space = Box(low=-np.inf, high=np.inf, shape=(336,), dtype=np.float32)
    act_space = Discrete(27)

    # Configuration matches train_ray_ppo_ma_shaped.py but adapted for inference
    config = {
        "env": "DummyMAEnv",
        "env_config": {
            "flatten_branched": True,
            "shaped_reward": True,
            "variation": EnvType.multiagent_player,
            # We don't need team_vs_random_multiagent since we override actions manually
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

    # Create the Trainers
    print("Creating PPO Trainer for Blue Team...")
    agent_blue = ppo.PPOTrainer(config=config, env="DummyMAEnv")
    print(f"Restoring Blue Team checkpoint from {checkpoint_path_blue}...")
    agent_blue.restore(checkpoint_path_blue)

    print("Creating PPO Trainer for Yellow Team...")
    agent_yellow = ppo.PPOTrainer(config=config, env="DummyMAEnv")
    print(f"Restoring Yellow Team checkpoint from {checkpoint_path_yellow}...")
    agent_yellow.restore(checkpoint_path_yellow)

    # Create the environment for visualization
    print("Creating visualization environment...")
    env_config_viz = config["env_config"].copy()
    env_config_viz["worker_id"] = random.randint(200, 500) # Avoid port conflicts
    env_config_viz["watch"] = True  # Enable watch mode for real-time visualization
    
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
                # Blue team vs Yellow team using different models
                for agent_id, agent_obs in obs.items():
                    if str(agent_id) in ["0", "1"]:
                        actions[agent_id] = agent_blue.compute_action(agent_obs)
                    else:
                        actions[agent_id] = agent_yellow.compute_action(agent_obs)
            else:
                # Handle stacked team observation (1 player acting for team = 2 x 336 dim)
                if len(obs.shape) == 1 and obs.shape[0] == 672:
                    obs_1 = obs[:336]
                    obs_2 = obs[336:]
                    a1 = agent_blue.compute_action(obs_1)
                    a2 = agent_blue.compute_action(obs_2)
                    actions = a1 * 27 + a2
                else:
                    actions = agent_blue.compute_action(obs)
            
            # Step the environment
            obs, reward, done, info = env.step(actions)

            # Accumulate rewards
            if isinstance(reward, dict):
                for agent_id, r in reward.items():
                    total_reward[agent_id] = total_reward.get(agent_id, 0) + r
            else:
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

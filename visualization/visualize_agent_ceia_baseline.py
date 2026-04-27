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

from ceia_baseline_agent.agent_ray import RayAgent

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
    # Absolute path to your trained trained multi-agent checkpoint file
    checkpoint_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "checkpoints", "PPO_SoccerShapedMASelfPlay_3e80a_00000_0_2026-03-01_17-16-08", "checkpoint_003500", "checkpoint-3500"))

    # Register the environments
    register_env("SoccerShapedMA", create_rllib_env)
    register_env("DummyMAEnv", lambda config: DummyMAEnv())

    # Initialize Ray
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    ray.init(ignore_reinit_error=True, num_gpus=0, include_dashboard=False)
    
    obs_space = Box(low=-np.inf, high=np.inf, shape=(336,), dtype=np.float32)
    act_space = Discrete(27)

    # Configuration for your trained agent
    config = {
        "env": "DummyMAEnv",
        "env_config": {
            "flatten_branched": True,
            "shaped_reward": True,
            "variation": EnvType.multiagent_player,
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
    print("Creating PPO Trainer for Blue Team (Your trained agent)...")
    agent = ppo.PPOTrainer(config=config, env="DummyMAEnv")

    # Restore the checkpoint
    print(f"Restoring checkpoint from {checkpoint_path}...")
    agent.restore(checkpoint_path)

    # Create the environment for visualization
    print("Creating visualization environment...")
    env_config_viz = config["env_config"].copy()
    env_config_viz["worker_id"] = random.randint(200, 500) # Avoid port conflicts
    env_config_viz["watch"] = True  # Enable watch mode for real-time visualization
    
    env = create_rllib_env(env_config_viz)

    # Initialize the baseline agent
    print("Loading CEIA Baseline Agent for Yellow Team...")
    # Pass a dummy wrapper as it might be checked briefly, although RayAgent generates its own env
    baseline_agent = RayAgent(env)

    print(f"Action Space: {env.action_space}")
    print("Starting visualization loop...")
    try:
        obs = env.reset()
        total_reward = {}
        while True:
            actions = {}
            if isinstance(obs, dict):
                agent_keys = list(obs.keys())
                blue_keys = [k for k in agent_keys if str(k) in ["0", "1"]]
                yellow_keys = [k for k in agent_keys if str(k) in ["2", "3"]]
                
                # Setup Blue team (our trained PPO policy)
                for agent_id in blue_keys:
                    actions[agent_id] = agent.compute_action(obs[agent_id])
                
                # Setup Yellow team (CEIA Baseline policy)
                baseline_obs = {y_key: obs[y_key] for y_key in yellow_keys}
                
                if baseline_obs:
                    # Act requires a dict, returns a dict
                    baseline_actions = baseline_agent.act(baseline_obs)
                    for y_key in yellow_keys:
                        if y_key in baseline_actions:
                            a = baseline_actions[y_key]
                            # Handle both Discrete and MultiDiscrete gracefully
                            if isinstance(a, (int, np.integer)) or np.isscalar(a) or np.ndim(a) == 0:
                                actions[y_key] = int(a)
                            else:
                                actions[y_key] = int(a[0]) * 9 + int(a[1]) * 3 + int(a[2])
                        else:
                            actions[y_key] = random.randint(0, 26) # Fallback

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

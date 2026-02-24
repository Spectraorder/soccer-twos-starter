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

from example_team_agent.agent import TeamAgent

def main():
    # Absolute path to the multi-agent checkpoint file
    checkpoint_path = r"d:\Georgia Tech\Second Year\Deep Reinforcement Learning\Final Project\github\soccer-twos-starter\checkpoints\PPO_SoccerShapedMABaseline_cfd23_00000_0_2026-02-23_16-51-31\checkpoint_001650\checkpoint-1650"

    # Register the environment
    register_env("SoccerShapedMA", create_rllib_env)

    # Initialize Ray
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
    env_config_viz["worker_id"] = random.randint(200, 500) # Avoid port conflicts
    env_config_viz["watch"] = True  # Enable watch mode for real-time visualization
    
    env_dummy = create_rllib_env(env_config_viz)
    
    # Create a FakeEnv so TeamAgent can read the unflattened action_space config it expects
    class FakeEnv:
        class ActionSpace:
            nvec = [3, 3, 3]
        class ObsSpace:
            shape = (336,)
        action_space = ActionSpace()
        observation_space = ObsSpace()

    # Initialize the baseline agent
    print("Loading Baseline TeamAgent...")
    baseline_agent = TeamAgent(FakeEnv())
    
    print(f"Action Space: {env_dummy.action_space}")
    print("Starting visualization loop...")
    try:
        obs = env_dummy.reset()
        total_reward = {}
        while True:
            # Helper to query the agents for actions
            actions = {}
            if isinstance(obs, dict):
                # Check mapping format
                agent_keys = list(obs.keys())
                blue_keys = [k for k in agent_keys if str(k) in ["0", "1"]]
                yellow_keys = [k for k in agent_keys if str(k) in ["2", "3"]]
                
                # Setup Blue team (our trained PPO policy)
                for agent_id in blue_keys:
                    actions[agent_id] = agent.compute_action(obs[agent_id])
                
                # Setup Yellow team (Baseline DQN policy)
                baseline_obs = {}
                for idx, y_key in enumerate(yellow_keys):
                    # We map yellow_keys back to 0, 1 for the baseline agent
                    baseline_obs[idx] = obs[y_key]
                
                if baseline_obs:
                    # Act requires a dict, returns a dict with MultiDiscrete actions (e.g., [1, 0, 2])
                    baseline_actions = baseline_agent.act(baseline_obs)
                    # Map back to the original environment IDs and manually flatten into Discrete(27)
                    for idx, y_key in enumerate(yellow_keys):
                        if idx in baseline_actions:
                            act_arr = baseline_actions[idx]
                            # Flatten MultiDiscrete [3, 3, 3] to Discrete(27)
                            flat_act = int(act_arr[0]) * 9 + int(act_arr[1]) * 3 + int(act_arr[2])
                            actions[y_key] = flat_act
                        else:
                            actions[y_key] = random.randint(0, 26) # Fallback

            # Step the environment
            obs, reward, done, info = env_dummy.step(actions)

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
                obs = env_dummy.reset()
    except KeyboardInterrupt:
        print("\nVisualization stopped by user.")
    finally:
        env_dummy.close()
        ray.shutdown()

if __name__ == "__main__":
    main()

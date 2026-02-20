import random
import ray
import ray.rllib.agents.ppo as ppo
from ray.tune.registry import register_env
import soccer_twos
from utils import create_rllib_env

import os

def main():
    # Absolute path to the checkpoint file
    checkpoint_path = r"d:\Georgia Tech\Second Year\Deep Reinforcement Learning\Final Project\github\soccer-twos-starter\checkpoints\PPO_SoccerShaped_bec43_00000_0_2026-02-19_16-48-37\checkpoint_000167\checkpoint-167"

    # Register the environment
    register_env("SoccerShaped", create_rllib_env)

    # Initialize Ray
    # Force Ray to ignore GPUs to avoid detection errors on Windows when nvidia-smi is missing
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    ray.init(ignore_reinit_error=True, num_gpus=0, include_dashboard=False)

    # Configuration matches params.json but adapted for inference
    config = {
        "env": "SoccerShaped",
        "env_config": {
            "flatten_branched": True,
            "multiagent": False,
            "num_envs_per_worker": 1,
            "opponent_policy": lambda *_: 0,  # Stationary opponent
            "shaped_reward": True,
            "single_player": True,
            "variation": soccer_twos.EnvType.team_vs_policy,
        },
        "framework": "torch",
        "model": {
            "fcnet_hiddens": [512],
            "vf_share_layers": True,
        },
        "num_gpus": 0,
        "num_workers": 0,  # Local worker
        "explore": False,
    }

    # Create the Trainer
    print("Creating PPO Trainer...")
    agent = ppo.PPOTrainer(config=config, env="SoccerShaped")

    # Restore the checkpoint
    print(f"Restoring checkpoint from {checkpoint_path}...")
    agent.restore(checkpoint_path)

    # Create the environment for visualization
    print("Creating environment...")
    env_config_viz = config["env_config"].copy()
    env_config_viz["worker_id"] = 200
    # env_config_viz["render"] = True
    # env_config_viz["time_scale"] = 1.0
    env_config_viz["watch"] = True  # Enable watch mode for real-time visualization
    env_config_viz["multiagent"] = True  # Enable multiagent mode to get dict observations
    
    # Set opponent policy to random (0-26 inclusive for 3x3x3 action space)
    env_config_viz["opponent_policy"] = lambda *_: random.randint(0, 26)
    env_config_viz["single_player"] = False  # Control both players on the blue team
    
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
                    # print(f"Agent {agent_id} obs shape: {agent_obs.shape}")
                    actions[agent_id] = agent.compute_action(agent_obs)
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

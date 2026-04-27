import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import ray
from ray import tune
from soccer_twos import EnvType
from utils import create_rllib_env
import os
import socket
import logging
import warnings

NUM_ENVS_PER_WORKER = 3
CHECKPOINT_PATH = "/home/hice1/ychen3868/DRL/github/soccer-twos-starter/ray_results_shaped/PPO_Shaped_MA_SelfPlay/PPO_SoccerShapedMASelfPlay_91151_00000_0_2026-02-26_11-59-52/checkpoint_003050/checkpoint-3050"


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    logging.getLogger("ray").setLevel(logging.ERROR)

    ray.init()

    tune.registry.register_env("SoccerShapedMASelfPlay", create_rllib_env)

    # Policy mapping function maps all agents (0, 1, 2, 3) to the "default_policy"
    # This means the PPO model weights will be shared and trained across all 4 players simultaneously
    def policy_mapping_fn(agent_id, *args, **kwargs):
        return "default_policy"

    # Define env_config
    env_config = {
        "num_envs_per_worker": NUM_ENVS_PER_WORKER,
        "variation": EnvType.multiagent_player, # True 4-player environment (dictionary mode)
        "flatten_branched": True,
        "shaped_reward": True,
        # Notice we REMOVE team_vs_random_multiagent! 
        # This allows utils.py to return all 4 agents correctly to Ray for Self-Play
    }

    # Instead of creating a dummy environment (which randomly crashes and blocks ports in Unity),
    # we manually define the known observation and action spaces for SoccerTwos.
    import gym
    from gym.spaces import Box, Discrete
    import numpy as np
    
    # SoccerTwos with flatten_branched=True has 336 obs and 1 discrete action of size 27 (3*3*3)
    obs_space = Box(low=-np.inf, high=np.inf, shape=(336,), dtype=np.float32)
    act_space = Discrete(27)

    analysis = tune.run(
        "PPO",
        name="PPO_Shaped_MA_SelfPlay",
        restore=CHECKPOINT_PATH,
        config={
            "num_gpus": 1,
            "num_workers": 8,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "log_level": "INFO",
            "framework": "torch",
            "env": "SoccerShapedMASelfPlay",
            "env_config": env_config,
            # Map all 4 players in Unity to the exact same shared PyTorch neural network
            "multiagent": {
                "policies": {"default_policy": (None, obs_space, act_space, {})},
                "policy_mapping_fn": policy_mapping_fn,
            },
            "model": {
                "vf_share_layers": True,
                "fcnet_hiddens": [512], # Must match the checkpoint architecture!
            },
            "rollout_fragment_length": 500,
            "train_batch_size": 12000,
        },
        stop={
            "timesteps_total": 60000000, # Go past the previous limits
        },
        checkpoint_freq=50,
        checkpoint_at_end=True,
        local_dir="./ray_results_shaped",
    )

    best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
    print(best_trial)
    best_checkpoint = analysis.get_best_checkpoint(
        trial=best_trial, metric="episode_reward_mean", mode="max"
    )
    print(best_checkpoint)
    print("Done multi-agent self-play training")

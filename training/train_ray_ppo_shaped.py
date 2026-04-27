import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import ray
from ray import tune
from soccer_twos import EnvType, make

from utils import create_rllib_env

# Use same config as example_ray_ppo_sp_still.py but add shaped_reward=True
NUM_ENVS_PER_WORKER = 3

if __name__ == "__main__":
    ray.init()

    tune.registry.register_env("SoccerShaped", create_rllib_env)

    analysis = tune.run(
        "PPO",
        name="PPO_Shaped",
        config={
            # system settings
            "num_gpus": 1,
            "num_workers": 8, # Use 4 workers to reduce load if needed, but keeping 8 as in example
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "log_level": "INFO",
            "framework": "torch",
            # RL setup
            "env": "SoccerShaped",
            "env_config": {
                "num_envs_per_worker": NUM_ENVS_PER_WORKER,
                "variation": EnvType.team_vs_policy,
                "multiagent": False,
                "single_player": True,
                "flatten_branched": True,
                "opponent_policy": lambda *_: 0,
                "shaped_reward": True, # Triggers our wrapper
            },
            "model": {
                "vf_share_layers": True,
                "fcnet_hiddens": [512],
            },
            "rollout_fragment_length": 500,
            "train_batch_size": 12000,
        },
        stop={
            "timesteps_total": 2000000, # 2M for quicker test, original was 20M
            # "time_total_s": 14400, # 4h
        },
        checkpoint_freq=50, # More frequent checkpoints for testing
        checkpoint_at_end=True,
        local_dir="./ray_results_shaped",
    )

    # Gets best trial based on max accuracy across all training iterations.
    best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
    print(best_trial)
    # Gets best checkpoint for trial based on accuracy.
    best_checkpoint = analysis.get_best_checkpoint(
        trial=best_trial, metric="episode_reward_mean", mode="max"
    )
    print(best_checkpoint)
    print("Done training")

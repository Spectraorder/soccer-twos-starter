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
CHECKPOINT_PATH = "/home/hice1/ychen3868/DRL/github/soccer-twos-starter/ray_results_shaped/PPO_Shaped_MA/PPO_SoccerShapedMA_2541d_00000_0_2026-02-21_13-57-19/checkpoint_001550/checkpoint-1550"

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    logging.getLogger("ray").setLevel(logging.ERROR)

    ray.init()

    tune.registry.register_env("SoccerShapedMABaseline", create_rllib_env)

    # Policy mapping function maps all agents to the "default_policy"
    # The checkpointed single-agent model weights are saved under "default_policy"
    def policy_mapping_fn(agent_id, *args, **kwargs):
        return "default_policy"

    # Define env_config
    env_config = {
        "num_envs_per_worker": NUM_ENVS_PER_WORKER,
        "variation": EnvType.multiagent_player, # True 4-player environment (dictionary mode)
        "flatten_branched": True,
        "shaped_reward": True,
        "team_vs_random_multiagent": True, # Our new wrapper in utils.py filters to 2vRandom
    }

    # To prevent Ray from trying to pickle the PyTorch Generator in TeamAgent's model,
    # we initialize the BaselineAgent lazily only when the policy is first called on the worker.
    class LazyBaselinePolicy:
        def __init__(self):
            self.baseline_agent = None

        def __call__(self, obs, *args):
            import torch
            import numpy as np
            if self.baseline_agent is None:
                from example_team_agent import TeamAgent
                class DummySpaceEnv:
                    @property
                    def action_space(self):
                        class _Act:
                            nvec = [3, 3, 3]
                        return _Act()
                        
                    @property
                    def observation_space(self):
                        class _Obs:
                            shape = (336,)
                        return _Obs()
                self.baseline_agent = TeamAgent(DummySpaceEnv())
                
            # TeamAgent.act() returns a MultiDiscrete list, but our Unity environment 
            # with flatten_branched=True expects a single Discrete integer index (0-26).
            # We bypass TeamAgent.act() and query its PyTorch model directly to get the raw integer.
            state = torch.from_numpy(obs).float().unsqueeze(0)
            action_values = self.baseline_agent.model(state)
            action = int(np.argmax(action_values.data.numpy()))
            
            return action

    env_config["opponent_policy"] = LazyBaselinePolicy()

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
        name="PPO_Shaped_MA_Baseline",
        restore=CHECKPOINT_PATH,
        config={
            "num_gpus": 1,
            "num_workers": 8,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "log_level": "INFO",
            "framework": "torch",
            "env": "SoccerShapedMABaseline",
            "env_config": env_config,
            # Map both players in the Team to the same shared policy model
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
            "timesteps_total": 45000000, # Go past the previous limit
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
    print("Done multi-agent training vs baseline")

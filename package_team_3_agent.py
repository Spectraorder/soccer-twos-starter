import os
import shutil

def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    agent_dir = os.path.join(root_dir, "team_3_agent")
    
    if not os.path.exists(agent_dir):
        os.makedirs(agent_dir)
        
    # Create __init__.py
    with open(os.path.join(agent_dir, "__init__.py"), "w") as f:
        f.write("\n")
        
    # Create README.md
    with open(os.path.join(agent_dir, "README.md"), "w") as f:
        f.write("# Team 3 Agent\n\n")
        f.write("**Agent name:** Team 3 PPOMultiagentSelfPlay\n\n")
        f.write("## Description\n\n")
        f.write("An agent trained with PPO via multi-agent self-play using Ray RLLib.\n")

    # Source checkpoint path
    source_run_dir = os.path.join(root_dir, r"checkpoints\PPO_SoccerShapedMASelfPlay_3e80a_00000_0_2026-03-01_17-16-08")
    source_checkpoint = os.path.join(source_run_dir, r"checkpoint_003500")
    
    # Target checkpoint path
    target_results_dir = os.path.join(agent_dir, "ray_results", "PPO_SoccerShapedMASelfPlay_3e80a_00000_0_2026-03-01_17-16-08")
    target_checkpoint = os.path.join(target_results_dir, "checkpoint_003500")
    
    # 1. Copy the run dir's params.pkl and similar to target run dir
    if not os.path.exists(target_results_dir):
        os.makedirs(target_results_dir)
    
    # We must copy params.pkl because RayAgent looks for it
    source_params = os.path.join(source_run_dir, "params.pkl")
    target_params = os.path.join(target_results_dir, "params.pkl")
    if os.path.exists(source_params):
        shutil.copy2(source_params, target_params)
        print(f"Copied {source_params} -> {target_params}")
        
    # 2. Copy the checkpoint dir
    if not os.path.exists(target_checkpoint) and os.path.exists(source_checkpoint):
        shutil.copytree(source_checkpoint, target_checkpoint)
        print(f"Copied checkpoint -> {target_checkpoint}")
        
    # Create agent_ray.py
    agent_code = """import pickle
import os
from typing import Dict

import gym
import numpy as np
import ray
from ray import tune
from gym.spaces import Box, Discrete
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import get_trainable_cls

from soccer_twos import AgentInterface

ALGORITHM = "PPO"
CHECKPOINT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "./ray_results/PPO_SoccerShapedMASelfPlay_3e80a_00000_0_2026-03-01_17-16-08/checkpoint_003500/checkpoint-3500",
)
POLICY_NAME = "default_policy"

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

class RayAgent(AgentInterface):
    def __init__(self, env: gym.Env):
        super().__init__()
        ray.init(ignore_reinit_error=True)

        config_path = ""
        if CHECKPOINT_PATH:
            config_dir = os.path.dirname(CHECKPOINT_PATH)
            config_path = os.path.join(config_dir, "params.pkl")
            if not os.path.exists(config_path):
                config_path = os.path.join(config_dir, "../params.pkl")

        if os.path.exists(config_path):
            with open(config_path, "rb") as f:
                config = pickle.load(f)
        else:
            raise ValueError(
                "Could not find params.pkl in either the checkpoint dir or "
                "its parent directory!"
            )

        config["num_workers"] = 0
        config["num_gpus"] = 0

        tune.registry.register_env("DummyMAEnv", lambda *_: DummyMAEnv())
        config["env"] = "DummyMAEnv"

        cls = get_trainable_cls(ALGORITHM)
        agent = cls(env=config["env"], config=config)
        agent.restore(CHECKPOINT_PATH)
        self.policy = agent.get_policy(POLICY_NAME)

    def act(self, observation: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        actions = {}
        for player_id in observation:
            actions[player_id], *_ = self.policy.compute_single_action(
                observation[player_id]
            )
        return actions
"""
    with open(os.path.join(agent_dir, "agent_ray.py"), "w") as f:
        f.write(agent_code)
    
    print(f"Successfully packaged Team 3 agent inside {agent_dir}")

if __name__ == "__main__":
    main()

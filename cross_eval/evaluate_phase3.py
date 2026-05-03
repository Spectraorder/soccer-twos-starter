import sys
import os
import random
import ray
import ray.rllib.agents.ppo as ppo
from ray.tune.registry import register_env
import numpy as np
from gym.spaces import Box, Discrete

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import soccer_twos
from utils import create_rllib_env

CHECKPOINT_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "checkpoints",
    "PPO_SoccerShapedMASelfPlay_3e80a_00000_0_2026-03-01_17-16-08",
    "checkpoint_003500", "checkpoint-3500"
))

class LazyBaselinePolicy:
    def __init__(self):
        self.baseline_agent = None
    def __call__(self, obs, *args):
        import torch
        if self.baseline_agent is None:
            from example_team_agent import TeamAgent
            class DummySpaceEnv:
                @property
                def action_space(self):
                    class _Act: nvec = [3, 3, 3]
                    return _Act()
                @property
                def observation_space(self):
                    class _Obs: shape = (336,)
                    return _Obs()
            self.baseline_agent = TeamAgent(DummySpaceEnv())
        state = torch.from_numpy(obs).float().unsqueeze(0)
        action_values = self.baseline_agent.model(state)
        return int(np.argmax(action_values.data.numpy()))

if __name__ == "__main__":
    agent_phase = "phase3"
    opponent_type = "baseline"
    num_episodes = 30

    print(f"\n\n--- Starting Evaluation: {agent_phase.upper()} Agent vs {opponent_type.upper()} Opponent for {num_episodes} episodes ---")
    print(f"Loading agent from: {CHECKPOINT_PATH}")

    ray.init(ignore_reinit_error=True, num_gpus=0, log_to_driver=False)
    register_env("SoccerShaped", create_rllib_env)

    env_config_eval = {
        "flatten_branched": True, "variation": soccer_twos.EnvType.multiagent_player,
        "team_vs_random_multiagent": True, "shaped_reward": False,
        "watch": False, "render": False, "time_scale": 20.0,
        "base_port": 50000 + random.randint(1000, 9000),
    }
    env_config_eval["opponent_policy"] = LazyBaselinePolicy()
    
    obs_space = Box(low=-np.inf, high=np.inf, shape=(336,), dtype=np.float32)
    act_space = Discrete(27)
    
    # 终极修复：
    config = {
        "env": "SoccerShaped", # 1. 提供 env 标识符
        "env_config": {"variation": soccer_twos.EnvType.multiagent_player}, # 2. 提供一个最小化的配置，以便能正确读取 space
        "create_env_on_driver": False, # 3. 【关键】禁止 PPOTrainer 在主进程中创建完整的环境实例
        "framework": "torch",
        "num_workers": 0,
        "num_gpus": 0,
        "model": {"fcnet_hiddens": [512], "vf_share_layers": True},
        "multiagent": {
            "policies": {"default_policy": (None, obs_space, act_space, {})},
            "policy_mapping_fn": lambda *_: "default_policy",
        },
    }

    agent = ppo.PPOTrainer(config=config)
    agent.restore(CHECKPOINT_PATH)
    
    env = create_rllib_env(env_config_eval)

    wins, losses, draws = 0, 0, 0
    try:
        for ep in range(num_episodes):
            obs = env.reset()
            done = False
            while not done:
                actions = {agent_id: agent.compute_action(agent_obs, policy_id="default_policy") for agent_id, agent_obs in obs.items()}
                obs, reward, d, info = env.step(actions)
                done = d["__all__"]
            
            team_score = sum(r for agent_id, r in reward.items() if agent_id in [0, 1])
            if team_score > 0: wins += 1
            elif team_score < 0: losses += 1
            else: draws += 1
            
            if (ep + 1) % 10 == 0:
                print(f"Episode {ep+1:3d}/{num_episodes} | Wins: {wins:3d} | Losses: {losses:3d} | Draws: {draws:3d}")
    finally:
        env.close()
        ray.shutdown()
        print("\n=====================================")
        print(f"FINAL RESULTS: {agent_phase.upper()} vs {opponent_type.upper()}")
        print(f"Win Rate:  {wins/num_episodes*100:.1f}% ({wins}/{num_episodes})")
        print("=====================================\n")

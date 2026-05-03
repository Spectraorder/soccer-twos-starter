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

# Checkpoint paths for the two models.
CHECKPOINT_PATHS = {
    "phase2": os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "checkpoints",
        "PPO_SoccerShapedMABaseline_cfd23_00000_0_2026-02-23_16-51-31",
        "checkpoint_001650", "checkpoint-1650"
    )),
    "phase3": os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "checkpoints",
        "PPO_SoccerShapedMASelfPlay_3e80a_00000_0_2026-03-01_17-16-08",
        "checkpoint_003500", "checkpoint-3500"
    )),
}

def make_dummy_env_config(worker_id, base_port):
    """
    Minimal Unity env config used only to satisfy RLlib trainer initialization.

    The trainer may still create a local worker in this older RLlib version, so each
    trainer receives a different worker_id/base_port pair to avoid Unity socket conflicts.
    These trainer-created envs are not used for evaluation rollouts.
    """
    return {
        "flatten_branched": True,
        "variation": soccer_twos.EnvType.multiagent_player,
        "team_vs_random_multiagent": True,
        "shaped_reward": False,
        "watch": False,
        "render": False,
        "time_scale": 20.0,
        "worker_id": worker_id,
        "base_port": base_port,
    }


def make_trainer_config(worker_id, base_port):
    """Create an RLlib PPOTrainer config matching the saved checkpoints."""
    obs_space = Box(low=-np.inf, high=np.inf, shape=(336,), dtype=np.float32)
    act_space = Discrete(27)

    return {
        "env": "SoccerShaped",
        "env_config": make_dummy_env_config(worker_id=worker_id, base_port=base_port),
        "create_env_on_driver": False,
        "framework": "torch",
        "num_workers": 0,
        "num_gpus": 0,
        "explore": False,
        "model": {"fcnet_hiddens": [512], "vf_share_layers": True},
        "multiagent": {
            "policies": {"default_policy": (None, obs_space, act_space, {})},
            "policy_mapping_fn": lambda *_: "default_policy",
        },
    }


class RLLibOpponentPolicy:
    """Callable opponent policy backed by an RLlib checkpoint."""

    def __init__(self, checkpoint_path, worker_id, base_port):
        config = make_trainer_config(worker_id=worker_id, base_port=base_port)
        self.agent = ppo.PPOTrainer(config=config, env="SoccerShaped")
        self.agent.restore(checkpoint_path)
        print("Opponent agent restored.")

    def __call__(self, obs, *args):
        return self.agent.compute_action(
            obs,
            policy_id="default_policy",
            explore=False,
        )

    def stop(self):
        self.agent.stop()


if __name__ == "__main__":
    num_episodes = 30
    our_agent_phase = "phase2"
    opponent_agent_phase = "phase3"

    print(
        f"\n\n--- Starting Evaluation: {our_agent_phase.upper()} Agent "
        f"vs {opponent_agent_phase.upper()} Opponent for {num_episodes} episodes ---"
    )

    ray.init(ignore_reinit_error=True, num_gpus=0, log_to_driver=False, include_dashboard=False)
    register_env("SoccerShaped", create_rllib_env)

    # Use three separated port/worker ranges:
    #   opponent trainer, our trainer, and the actual evaluation environment.
    port_seed = 50000 + random.randint(1000, 7000)
    opponent_base_port = port_seed
    our_base_port = port_seed + 1000
    eval_base_port = port_seed + 2000

    env = None
    opponent_policy = None
    our_agent = None

    try:
        # Load the opponent checkpoint.
        print(f"Loading opponent agent from: {CHECKPOINT_PATHS[opponent_agent_phase]}")
        opponent_policy = RLLibOpponentPolicy(
            CHECKPOINT_PATHS[opponent_agent_phase],
            worker_id=10,
            base_port=opponent_base_port,
        )

        # Load our checkpoint.
        print(f"Loading our agent from: {CHECKPOINT_PATHS[our_agent_phase]}")
        our_config = make_trainer_config(worker_id=20, base_port=our_base_port)
        our_agent = ppo.PPOTrainer(config=our_config, env="SoccerShaped")
        our_agent.restore(CHECKPOINT_PATHS[our_agent_phase])

        # Create the only environment used for the actual matches.
        env_config_eval = {
            "flatten_branched": True,
            "variation": soccer_twos.EnvType.multiagent_player,
            "team_vs_random_multiagent": True,
            "shaped_reward": False,
            "watch": False,
            "render": False,
            "time_scale": 20.0,
            "worker_id": 30,
            "base_port": eval_base_port,
            "opponent_policy": opponent_policy,
        }
        env = create_rllib_env(env_config_eval)

        wins, losses, draws = 0, 0, 0
        goal_diffs = []
        episode_returns = []

        for ep in range(num_episodes):
            obs = env.reset()
            done = False
            episode_return = 0.0

            while not done:
                actions = {
                    agent_id: our_agent.compute_action(
                        agent_obs,
                        policy_id="default_policy",
                        explore=False,
                    )
                    for agent_id, agent_obs in obs.items()
                }
                obs, reward, d, info = env.step(actions)

                # With shaped_reward=False, this is closer to game outcome than shaped reward.
                step_team_reward = sum(
                    r for agent_id, r in reward.items()
                    if agent_id in [0, 1]
                )
                episode_return += step_team_reward
                done = d["__all__"]

            episode_returns.append(episode_return)
            goal_diffs.append(episode_return)

            if episode_return > 0:
                wins += 1
            elif episode_return < 0:
                losses += 1
            else:
                draws += 1

            print(
                f"Episode {ep + 1:3d}/{num_episodes} | "
                f"Return: {episode_return:7.3f} | "
                f"Wins: {wins:3d} | Losses: {losses:3d} | Draws: {draws:3d}"
            )

    finally:
        if env is not None:
            env.close()
        if our_agent is not None:
            our_agent.stop()
        if opponent_policy is not None:
            opponent_policy.stop()
        ray.shutdown()

        completed = max(1, wins + losses + draws)
        avg_return = float(np.mean(episode_returns)) if episode_returns else 0.0
        print("\n=====================================")
        print(f"FINAL RESULTS: {our_agent_phase.upper()} vs {opponent_agent_phase.upper()}")
        print(f"Episodes:  {completed}")
        print(f"Wins:      {wins}")
        print(f"Losses:    {losses}")
        print(f"Draws:     {draws}")
        print(f"Win Rate:  {wins / completed * 100:.1f}% ({wins}/{completed})")
        print(f"Avg Return:{avg_return:.3f}")
        print("=====================================\n")

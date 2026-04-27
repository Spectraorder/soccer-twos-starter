from random import uniform as randfloat

import gym
from ray.rllib import MultiAgentEnv
import soccer_twos


class RLLibWrapper(gym.core.Wrapper, MultiAgentEnv):
    """
    A RLLib wrapper so our env can inherit from MultiAgentEnv.
    """
    pass

class MultiAgentTeamVsPolicyWrapper(gym.core.Wrapper, MultiAgentEnv):
    """
    Wraps the standard 4-player EnvType.multiagent_player environment.
    Exposes Blue Team (agents 0, 1) as independent agents for RLlib MultiAgentEnv.
    Applies an opponent policy (default random) to Orange Team (agents 2, 3).
    """
    def __init__(self, env, opponent_policy=None):
        super().__init__(env)
        if opponent_policy is None:
            self.opponent_policy = lambda *_: self.env.action_space.sample()
        else:
            self.opponent_policy = opponent_policy
            
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self._agent_ids = {0, 1}
        self.last_obs = None
        
    def reset(self):
        obs = self.env.reset()
        self.last_obs = obs
        return {0: obs[0], 1: obs[1]}
        
    def step(self, action_dict):
        env_action = {
            0: action_dict.get(0, self.env.action_space.sample()),
            1: action_dict.get(1, self.env.action_space.sample()),
            2: self.opponent_policy(self.last_obs[2]),
            3: self.opponent_policy(self.last_obs[3])
        }
        
        obs, reward, done, info = self.env.step(env_action)
        self.last_obs = obs
        
        out_obs = {0: obs[0], 1: obs[1]}
        out_reward = {0: reward[0], 1: reward[1]}
        
        # In multiagent_player variation, done is a dict with "__all__"
        is_done = done.get("__all__", False)
        out_done = {0: is_done, 1: is_done, "__all__": is_done}
        out_info = {0: info[0], 1: info[1]}
        
        return out_obs, out_reward, out_done, out_info


def create_rllib_env(env_config: dict = {}):
    """
    Creates a RLLib environment and prepares it to be instantiated by Ray workers.
    Args:
        env_config: configuration for the environment.
            You may specify the following keys:
            - variation: one of soccer_twos.EnvType. Defaults to EnvType.multiagent_player.
            - opponent_policy: a Callable for your agent to train against. Defaults to a random policy.
    """
    if hasattr(env_config, "worker_index"):
        env_config["worker_id"] = (
            env_config.worker_index * env_config.get("num_envs_per_worker", 1)
            + env_config.vector_index
        )
    env = soccer_twos.make(**env_config)
    # env = TransitionRecorderWrapper(env)
    if "shaped_reward" in env_config and env_config["shaped_reward"]:
        from env_wrapper import RewardShapedWrapper
        env = RewardShapedWrapper(env)

    if "multiagent" in env_config and not env_config["multiagent"]:
        # is multiagent by default, is only disabled if explicitly set to False
        return env
        
    # Check if we should wrap it for 2vPolicy MultiAgent
    if env_config.get("team_vs_random_multiagent", False):
        return MultiAgentTeamVsPolicyWrapper(env, opponent_policy=env_config.get("opponent_policy"))
        
    return RLLibWrapper(env)


def sample_vec(range_dict):
    return [
        randfloat(range_dict["x"][0], range_dict["x"][1]),
        randfloat(range_dict["y"][0], range_dict["y"][1]),
    ]


def sample_val(range_tpl):
    return randfloat(range_tpl[0], range_tpl[1])


def sample_pos_vel(range_dict):
    _s = {}
    if "position" in range_dict:
        _s["position"] = sample_vec(range_dict["position"])
    if "velocity" in range_dict:
        _s["velocity"] = sample_vec(range_dict["velocity"])
    return _s


def sample_player(range_dict):
    _s = sample_pos_vel(range_dict)
    if "rotation_y" in range_dict:
        _s["rotation_y"] = sample_val(range_dict["rotation_y"])
    return _s

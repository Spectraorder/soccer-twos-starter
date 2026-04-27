import gym
import numpy as np

class RewardShapedWrapper(gym.Wrapper):
    """
    A wrapper that adds dense rewards to the SoccerTwos environment.
    Supports both Single Agent (gym.Env) and Multi Agent (Ray MultiAgentEnv).
    """
    def __init__(self, env):
        super().__init__(env)
        self.last_ball_pos = None
        self.last_player_pos = {} # Keyed by agent_id for MA, or 'single_player' for SA

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.last_ball_pos = None
        self.last_player_pos = {}
        return obs

    def step(self, action):
        obs, rewards, done, info = self.env.step(action)
        
        # Check if Single Agent or Multi Agent
        is_multiagent = isinstance(rewards, dict)
        
        # Common Data Extraction
        current_ball_pos = None
        
        if is_multiagent:
            # Multi-agent logic
            # info is dict {agent_id: info_dict}
            if info:
                first_key = next(iter(info))
                if 'ball_info' in info[first_key]:
                    current_ball_pos = info[first_key]['ball_info']['position']
        else:
            # Single-agent logic
            # info is info_dict directly
            if 'ball_info' in info:
                current_ball_pos = info['ball_info']['position']

        # Calculate Rewards if we have ball info
        if current_ball_pos is not None:
            
            # --- Ball Movement Reward ---
            # Assume Team 0 attacks +X
            ball_dx_reward = 0.0
            if self.last_ball_pos is not None:
                dx = current_ball_pos[0] - self.last_ball_pos[0]
                ball_dx_reward = dx * 1.0 # Scale
            
            if is_multiagent:
                for pid in rewards:
                    if pid not in info:
                        continue
                    p_info = info[pid]
                    current_player_pos = p_info['player_info']['position']
                    
                    # Distance Reward
                    dist_reward = 0.0
                    if pid in self.last_player_pos and self.last_ball_pos is not None:
                        curr_dist = np.linalg.norm(current_player_pos - current_ball_pos)
                        prev_dist = np.linalg.norm(self.last_player_pos[pid] - self.last_ball_pos)
                        dist_reward = (prev_dist - curr_dist) * 1.0
                    
                    # Team Reward
                    team_ball_reward = 0.0
                    # Heuristic for team ID: 0,1 are Team 0. 2,3 are Team 1.
                    # Or we simply assume if we are controlling it, we want it to go to our goal.
                    # Team 0 attacks +X.
                    if pid < 2: 
                        team_ball_reward = ball_dx_reward
                    else:
                        team_ball_reward = -ball_dx_reward

                    rewards[pid] += (dist_reward + team_ball_reward)
                    self.last_player_pos[pid] = current_player_pos
                    
            else:
                # Single Agent
                # We assume we are Team 0 (Player 0 usually)
                if 'player_info' in info:
                    current_player_pos = info['player_info']['position']
                    
                    # Distance Reward
                    dist_reward = 0.0
                    if 'single' in self.last_player_pos and self.last_ball_pos is not None:
                        curr_dist = np.linalg.norm(current_player_pos - current_ball_pos)
                        prev_dist = np.linalg.norm(self.last_player_pos['single'] - self.last_ball_pos)
                        dist_reward = (prev_dist - curr_dist) * 1.0
                        
                    # Team Reward
                    # In single player team_vs_policy, we are Team 0 attacking +X
                    team_ball_reward = ball_dx_reward
                    
                    rewards += (dist_reward + team_ball_reward)
                    self.last_player_pos['single'] = current_player_pos

            # Update last ball pos
            self.last_ball_pos = current_ball_pos

        return obs, rewards, done, info

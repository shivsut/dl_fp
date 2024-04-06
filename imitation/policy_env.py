from stable_baselines3.common.policies import BasePolicy
# import pystk
import numpy as np
from gymnasium import spaces
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union
from os import path 
import torch 
# TRACK_NAME = 'icy_soccer_field'

class IceHockeyEnv(BasePolicy):
    """The base policy object.

    Parameters are mostly the same as `BaseModel`; additions are documented below.

    :param args: positional arguments passed through to `BaseModel`.
    :param kwargs: keyword arguments passed through to `BaseModel`.
    :param squash_output: For continuous actions, whether the output is squashed
        or not using a ``tanh()`` function.
    """

    def __init__(self,
                 observation_space: spaces.Space,
                action_space: spaces.Space, 
                num_players: int = 2, team: int = 0):
        super().__init__(
            observation_space,
            action_space)
        
        self.num_players = num_players
        self.team = team
        self.model = torch.jit.load(path.join(path.dirname(path.abspath(__file__)), 'jurgen_agent.pt'))
        
    # def extract_featuresV2(self, pstate, soccer_state, opponent_state, team_id):
    #     def limit_period(angle):
    #         # turn angle into -1 to 1 
    #         return angle - torch.floor(angle / 2 + 0.5) * 2 
        
    #     # features of ego-vehicle
    #     kart_front = torch.tensor(pstate['kart']['front'], dtype=torch.float32)[[0, 2]]
    #     kart_center = torch.tensor(pstate['kart']['location'], dtype=torch.float32)[[0, 2]]
    #     kart_direction = (kart_front-kart_center) / torch.norm(kart_front-kart_center)
    #     kart_angle = torch.atan2(kart_direction[1], kart_direction[0])

    #     # features of soccer 
    #     puck_center = torch.tensor(soccer_state['ball']['location'], dtype=torch.float32)[[0, 2]]
    #     kart_to_puck_direction = (puck_center - kart_center) / torch.norm(puck_center-kart_center)
    #     kart_to_puck_angle = torch.atan2(kart_to_puck_direction[1], kart_to_puck_direction[0]) 

    #     kart_to_puck_angle_difference = limit_period((kart_angle - kart_to_puck_angle)/np.pi)

    #     # features of score-line 
    #     goal_line_center = torch.tensor(soccer_state['goal_line'][(team_id+1)%2], dtype=torch.float32)[:, [0, 2]].mean(dim=0)

    #     puck_to_goal_line = (goal_line_center-puck_center) / torch.norm(goal_line_center-puck_center)

    #     features = torch.tensor([kart_center[0], kart_center[1], kart_angle, kart_to_puck_angle, 
    #         goal_line_center[0], goal_line_center[1], kart_to_puck_angle_difference, 
    #         puck_center[0], puck_center[1], puck_to_goal_line[0], puck_to_goal_line[1]], dtype=torch.float32)

    #     return features 

    # def act(self, player_state, opponent_state, soccer_state):
    #     actions = [] 
    #     for player_id, pstate in enumerate(player_state):
    #         features = self.extract_featuresV2(pstate, soccer_state, opponent_state, self.team)
    #         acceleration, steer, brake = self.model(features)
    #         actions.append(dict(acceleration=acceleration, steer=steer, brake=brake))                        
    #     return actions 

    # def obs_convert(self):

    #     pass 

    def _predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:

        # player_state, opponent_state, soccer_state = self.obs_convert(observation)
        # actions = self.act(player_state, opponent_state, soccer_state)
        # return actions, state 
        return self.model(observation), state
        



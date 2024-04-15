from stable_baselines3.common.policies import BasePolicy
import numpy as np
from gymnasium import spaces
from typing import Any, Dict, List, Optional, Tuple, Union
from os import path 
import torch

from imitation_agent.utils import discretization


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
                expert_name: str,
                num_players: int = 2, team: int = 0,
                 args=None):
        super().__init__(
            observation_space,
            action_space)
        
        self.num_players = num_players
        self.team = team
        self.model = torch.jit.load(_restore_shapes=True, f=path.join(path.dirname(path.abspath(__file__)), f'experts/{expert_name}.pt'))
        self.discrete = discretization(aceel_div=args.md) if args.md else lambda x :x

    def _predict(self, observation, deterministic: bool = False):
        actions = self.model(observation)
        return torch.tensor(actions)

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:

        # Convert the observation to tensor
        observation = torch.tensor(observation, dtype=torch.float32)
        # TODO optimise
        actions = []
        for i in range(len(observation)):
            obs = observation[i]

            actions.append(self.discrete(self._predict(obs)))
        return actions, state
        
        



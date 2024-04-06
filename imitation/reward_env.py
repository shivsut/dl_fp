import gymnasium
import numpy as np
import pystk
from gymnasium import spaces

import logging
import numpy as np
from collections import namedtuple
from ..tournament.utils import VideoRecorder

TRACK_NAME = 'icy_soccer_field'
MAX_FRAMES = 1000

RunnerInfo = namedtuple('RunnerInfo', ['agent_type', 'error', 'total_act_time'])

import gym
import torch
import pystk
import numpy as np

def to_native(o):
    # Super obnoxious way to hide pystk
    import pystk
    _type_map = {pystk.Camera.Mode: int,
                 pystk.Attachment.Type: int,
                 pystk.Powerup.Type: int,
                 float: float,
                 int: int,
                 list: list,
                 bool: bool,
                 str: str,
                 memoryview: np.array,
                 property: lambda x: None}

    def _to(v):
        if type(v) in _type_map:
            return _type_map[type(v)](v)
        else:
            return {k: _to(getattr(v, k)) for k in dir(v) if k[0] != '_'}
    return _to(o)


class IceHockeyEnvImitation(gymnasium.Env):
    """
    Observation:
        Image of shape `self.observation_shape`.

    Actions:
        -----------------------------------------------------------------
        |         ACTIONS               |       POSSIBLE VALUES         |
        -----------------------------------------------------------------
        |       Acceleration            |           (0, 1)              |
        |       Brake                   |           (0, 1)              |
        |       Drift                   |           (0, 1)              |
        |       Fire                    |           (0, 1)              |
        |       Nitro                   |           (0, 1)              |
        |       Rescue                  |           (0, 1)              |
        |       Steer                   |         (-1, 0, 1)            |
        -----------------------------------------------------------------
    """

    def __init__(self, args, logging_level=None):
        super(IceHockeyEnvImitation, self).__init__()
        self.do_init = True
        self.args = args
        self.logging_level = logging_level
        self.init()

    def init(self):
        if not self.do_init:
            return
        self.do_init = False
        self._pystk = pystk
        self._pystk.init(self._pystk.GraphicsConfig.none())
        if self.logging_level is not None:
            logging.basicConfig(level=self.logging_level)
        self.recorder = VideoRecorder(self.args.record_fn) if self.args.record_fn else None
        # self.action_space = spaces.MultiDiscrete(nvec=[10, 200, 2, 2, 2], start=[0, -100, 0, 0, 0])
        # self.action_space = spaces.Box(low=np.array([0, -1, 0, 0, 0]), high=np.array([1, 1, 0.1, 0.1, 0.1]), dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([0, -1,]), high=np.array([1, 1,]), dtype=np.float32)
        # self.action_space = spaces.Dict(accel_and_steer = spaces.Box(low=np.array([0, -1]), high=np.array([1, 1]), dtype=np.float32),brake = spaces.Discrete(n = 2, start=0))
        # TODO Max distance
        # [kart_to_puck_dist, alignment, goal_dist, puck_and_goal_distance]
        self.observation_space = spaces.Box(low=np.array([0, -1, 0, 0]), high=np.array([100, 1, 100, 100]), dtype=np.float32)

        # self.team1 = AIRunner() if kwargs['team1'] == 'AI' else TeamRunner("")
        # self.team2 = AIRunner() if kwargs['team2'] == 'AI' else TeamRunner("")
        self.timeout = 1e10
        self.max_score = 3
        self.num_players = 1
        # self.team1 = DummyTeam(self.num_players, 0)

        self.info = {}
        self.race_config = self._pystk.RaceConfig(track=TRACK_NAME, mode=self._pystk.RaceConfig.RaceMode.SOCCER, num_kart=1 * self.num_players)
        self.race_config.players.pop()
        for i in range(self.num_players):
            # self.race_config.players.append(self._make_config(0, hasattr(self.team1, 'is_ai') and self.team1.is_ai, ['tux']))
            self.race_config.players.append(self._make_config(0, False, 'tux'))
        # self.reset()
        self.race = self._pystk.Race(self.race_config)

    def _make_config(self, team_id, is_ai, kart):
        # TODO if not AI
        PlayerConfig = self._pystk.PlayerConfig
        controller = PlayerConfig.Controller.AI_CONTROL if is_ai else PlayerConfig.Controller.PLAYER_CONTROL
        return PlayerConfig(controller=controller, team=team_id, kart=kart)

    def step(self, action):
        #pystk
        team1_state = [to_native(p) for p in self.state.players[0::2]]
        team2_state = [to_native(p) for p in self.state.players[1::2]]
        soccer_state = to_native(self.state.soccer)
        logging.info('calling agent')
        team1_actions = self.team1.act(action)
        # team2_actions = self.team2.act(team2_state, team1_state, soccer_state)

        # TODO check for error in info and raise MatchException
        # TODO check for timeout

        if self.recorder:
            self.recorder(team1_state, team2_state, soccer_state=soccer_state, actions=team1_actions,team1_images=None, team2_images=None)

        if (not self.race.step([self._pystk.Action(**a) for a in team1_actions]) and self.num_players):
            self.truncated = True
        if (sum(self.state.soccer.score) >= self.max_score):
            self.terminated = True
        if not (self.truncated or self.terminated):
            self.state.update()

        logging.info('state updated, calculating reward')
        team1_state_next = [to_native(p) for p in self.state.players[0::2]]
        team2_state_next = [to_native(p) for p in self.state.players[1::2]]
        soccer_state = to_native(self.state.soccer)
        # p_features = extract_state_train(team1_state_next, team2_state_next, soccer_state, 0)

        reward = self.reward.step(p_features)
        logging.info(f'returning new state and reward {reward}')
        # print(f"reward: {reward}")
        return np.array(p_features), reward, self.terminated, self.truncated, self.info

    def reset(self, seed=1, options=None):
        self.init()
        # super().reset(seed=seed)
        logging.info('Resetting')
        # self.recorder = VideoRecorder('infer.mp4') if self.args.record_fn else None
        self.reward = Reward()
        self.truncated = False
        self.terminated = False
        logging.info('Starting new race')
        self.race.stop()
        self.race.start()
        self.state = self._pystk.WorldState()
        self.state.update() # TODO need to call this here?

        team1_state_next = [to_native(p) for p in self.state.players[0::2]]
        team2_state_next = [to_native(p) for p in self.state.players[1::2]]
        soccer_state = to_native(self.state.soccer)
        p_features = extract_state_train(team1_state_next, team2_state_next, soccer_state, 0)
        print(p_features)
        return np.array(p_features), self.info

    def close(self):
        self.race.stop()
        del self.race
        self._pystk.clean()



class AIRunner:
    agent_type = 'state'
    is_ai = True

    def new_match(self, team: int, num_players: int) -> list:
        pass

    def act(self, player_state, opponent_state, world_state):
        return []

    def info(self):
        return RunnerInfo('state', None, 0)
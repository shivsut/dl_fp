import gymnasium
import numpy as np
import pystk
from gymnasium import spaces

import logging
import numpy as np
from collections import namedtuple
from .utils import VideoRecorder

TRACK_NAME = 'icy_soccer_field'
MAX_FRAMES = 1000

RunnerInfo = namedtuple('RunnerInfo', ['agent_type', 'error', 'total_act_time'])

import gym
import torch
import pystk
import numpy as np
from .reward import Reward
from .extract_state import extract_state_train_p1, extract_state_train
from .extract_state import extract_state_train_p2

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


# class IceHockeyAgent:
#     """
#     SuperTuxKart agent for handling actions and getting state information from the environment.
#     The `STKEnv` class passes on the actions to this class for it to handle and gets the current
#     state(image) and various other info.
#
#     :param graphicConfig: `pystk.GraphicsConfig` object specifying various graphic configs
#     :param raceConfig: `pystk.RaceConfig` object specifying the track and other configs
#     """
#
#     def __init__(self, logging_level=None, **kwargs):
#         # Copied from runner.py
#         import pystk
#         self._pystk = pystk
#         graphics_config = self._pystk.GraphicsConfig.none()
#         self._pystk.init(graphics_config)
#         if logging_level is not None:
#             logging.basicConfig(level=logging_level)
#         # TODO: Define the following parameter
#         # RaceConfig = self._pystk.RaceConfig
#         self.timeout = 1e10
#         self.num_player = 2
#         self.initial_ball_location=[0, 0]
#         self.initial_ball_velocity=[0, 0]
#         self.max_score=3
#         self.race = None
#         self.state = None
#         self.record_fn = kwargs['record_fn']
#         self.team1 = kwargs['team1'] # AIRunner() if args.team1 == 'AI' else TeamRunner(args.team1)
#         self.team2 = kwargs['team2'] # AIRunner() if args.team2 == 'AI' else TeamRunner(args.team2)
#         # Irrelevant
#         self.started = False
#         self.AI = None
#         self.info = {}
#         self.reset()
#
#     def __del__(self):
#         if hasattr(self, '_pystk') and self._pystk is not None and self._pystk.clean is not None:  # Don't ask why...
#             self._pystk.clean()
#
#     def _make_config(self, team_id, is_ai, kart):
#         PlayerConfig = self._pystk.PlayerConfig
#         controller = PlayerConfig.Controller.AI_CONTROL if is_ai else PlayerConfig.Controller.PLAYER_CONTROL
#         return PlayerConfig(controller=controller, team=team_id, kart=kart)
#
#     @classmethod
#     def _r(cls, f):
#         if hasattr(f, 'remote'):
#             return f.remote
#         if hasattr(f, '__call__'):
#             if hasattr(f.__call__, 'remote'):
#                 return f.__call__.remote
#         return f
#
#     @staticmethod
#     def _g(f):
#         # from .remote import ray
#         # #if ray is not None and isinstance(f, (ray.types.ObjectRef, ray._raylet.ObjectRef)):
#         # if ray is not None and isinstance(f, (ray.ObjectRef, ray._raylet.ObjectRef)):
#         #     return ray.get(f)
#         return f
#
#     def _check(self, team1, team2, where, timeout):
#         _, error, t1 = self._g(self._r(team1.info)())
#         if error:
#             raise MatchException([0, 3], 'other team crashed', 'crash during {}: {}'.format(where, error))
#
#         _, error, t2 = self._g(self._r(team2.info)())
#         if error:
#             raise MatchException([3, 0], 'crash during {}: {}'.format(where, error), 'other team crashed')
#
#         logging.debug('timeout {} <? {} {}'.format(timeout, t1, t2))
#         return t1 < timeout, t2 < timeout
#
#     # TODO: implement a function to obtain the observation_space
#     # def get_observation(self, action: list):
#
#
#     def get_env_info(self) -> dict:
#         info = {}
#         return info
#
#     def get_info(self) -> dict:
#         info = {}
#         return info
#
#     def done(self) -> bool:
#         """
#         `playerKart.finish_time` > 0 when the kart finishes the race.
#         Initially the finish time is < 0.
#         """
#         return self.playerKart.finish_time > 0
#
#     def reset(self, seed=None):
#         self.truncated = False
#         self.terminated = False
#         RaceConfig = self._pystk.RaceConfig
#
#         logging.info('Creating teams')
#
#         # Start a new match
#         self.t1_cars = self._g(self._r(self.team1.new_match)(0, self.num_player)) or ['tux']
#         self.t2_cars = self._g(self._r(self.team2.new_match)(1, self.num_player)) or ['tux']
#
#         # Deal with crashes
#         self.t1_can_act, self.t2_can_act = self._check(self.team1, self.team2, 'new_match', self.timeout)
#
#         # Setup the race config
#         logging.info('Setting up race')
#
#         race_config = RaceConfig(track=TRACK_NAME, mode=RaceConfig.RaceMode.SOCCER, num_kart=2 * self.num_player)
#         race_config.players.pop()
#         for i in range(self.num_player):
#             race_config.players.append(self._make_config(0, hasattr(self.team1, 'is_ai') and self.team1.is_ai, self.t1_cars[i % len(self.t1_cars)]))
#             race_config.players.append(self._make_config(1, hasattr(self.team2, 'is_ai') and self.team2.is_ai, self.t2_cars[i % len(self.t2_cars)]))
#
#         # Start the match
#         logging.info('Starting race')
#         self.race = self._pystk.Race(race_config)
#         self.race.start()
#
#         self.state = self._pystk.WorldState()
#         self.state.update()
#         self.state.set_ball_location((self.initial_ball_location[0], 1, self.initial_ball_location[1]),
#                                 (self.initial_ball_velocity[0], 0, self.initial_ball_velocity[1]))
#
#         # self.close()
#
#
#     def step(self, action=None):
#         # self.state.update()
#         # Get the state
#         team1_state = [to_native(p) for p in self.state.players[0::2]]
#         team2_state = [to_native(p) for p in self.state.players[1::2]]
#         soccer_state = to_native(self.state.soccer)
#         # import pdb; pdb.set_trace()
#         team1_images = team2_images = None
#
#         # Play the match (given the states)
#         team1_actions_delayed = self._r(self.team1.act)(team1_state, team2_state, soccer_state)
#         team2_actions_delayed = self._r(self.team2.act)(team2_state, team1_state, soccer_state)
#
#         # Wait for the actions to finish
#         team1_actions = self._g(team1_actions_delayed) if self.t1_can_act else None
#         team2_actions = self._g(team2_actions_delayed) if self.t2_can_act else None
#
#         new_t1_can_act, new_t2_can_act = self._check(self.team1, self.team2, 'act', self.timeout)
#         if not new_t1_can_act and self.t1_can_act and self.verbose:
#             print('Team 1 timed out')
#         if not new_t2_can_act and self.t2_can_act and self.verbose:
#             print('Team 2 timed out')
#
#         self.t1_can_act, self.t2_can_act = new_t1_can_act, new_t2_can_act
#
#         # Assemble the actions
#         actions = []
#         for i in range(self.num_player):
#             a1 = team1_actions[i] if team1_actions is not None and i < len(team1_actions) else {}
#             a2 = team2_actions[i] if team2_actions is not None and i < len(team2_actions) else {}
#             actions.append(a1)
#             actions.append(a2)
#
#         if self.record_fn:
#             self._r(self.record_fn)(team1_state, team2_state, soccer_state=soccer_state, actions=actions,
#                                 team1_images=team1_images, team2_images=team2_images)
#
#         logging.debug('  race.step  [score = {}]'.format(self.state.soccer.score))
#
#         if (not self.race.step([self._pystk.Action(**a) for a in actions]) and self.num_player):
#             self.truncated = True
#         if (sum(self.state.soccer.score) >= self.max_score):
#             self.terminated = True
#         if not (self.truncated or self.terminated):
#             self.state.update()
#
#         self.observation = self.getObservation(team1_state)
#         self.reward = self.getReward(self.state.soccer.score, self.observation)
#         return self.observation, self.reward, self.terminated, self.truncated, self.info
#
#     def getObservation(self, state):
#         # dummy - team1 state
#         # import pdb; pdb.set_trace()
#         # return np.array(state)
#         return np.array([1.0]).astype(np.float32)
#
#     def getReward(self, scores, observation):
#         # Team1: Ours (positive reward)
#         # Team2: Opponent (negative reward)
#         return (scores[0] - scores[1])
#
#
#     def close(self):
#         self.race.stop()
#         del self.race
#         pystk.clean()


class DummyTeam():
    def __init__(self, num_players, team_id):
        self.num_players = num_players
        self.team_id = team_id
    def new_match(self):
        return ['tux'] * self.num_players
    def act(self, action):
        return [dict(acceleration=action[0], steer=action[1], brake=True if action[2] > 0.05 else False, nitro=True if action[3] > 0.05 else False, drift=True if action[4] > 0.05 else False, rescue=False, fire=False)]
        # return [dict(acceleration=action[0]/10, steer=action[1]/100, brake=True if action[2] >0 else False, nitro=True if action[3] >0 else False, drift=True if action[4] >0 else False, rescue=False, fire=False)]
    def reset(self):
        pass

class IceHockeyEnv(gymnasium.Env):
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
        super(IceHockeyEnv, self).__init__()
        self.args = args
        self._pystk = pystk
        self._pystk.init(self._pystk.GraphicsConfig.none())
        if logging_level is not None:
            logging.basicConfig(level=logging_level)
        self.recorder = VideoRecorder(args.record_fn) if args.record_fn else None
        # self.action_space = spaces.MultiDiscrete(nvec=[10, 200, 2, 2, 2], start=[0, -100, 0, 0, 0])
        self.action_space = spaces.Box(low=np.array([0, -1, 0, 0, 0]), high=np.array([1, 1, 0.1, 0.1, 0.1]), dtype=np.float32)
        # self.action_space = spaces.Dict(accel_and_steer = spaces.Box(low=np.array([0, -1]), high=np.array([1, 1]), dtype=np.float32),brake = spaces.Discrete(n = 2, start=0))
        # TODO Max distance
        self.observation_space = spaces.Box(low=np.array([0]), high=np.array([90]), dtype=np.float32)

        # self.team1 = AIRunner() if kwargs['team1'] == 'AI' else TeamRunner("")
        # self.team2 = AIRunner() if kwargs['team2'] == 'AI' else TeamRunner("")
        self.timeout = 1e10
        self.max_score = 3
        self.num_players = 1
        self.team1 = DummyTeam(self.num_players, 0)

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
        p_features = extract_state_train(team1_state_next, team2_state_next, soccer_state, 0)

        reward = self.reward.step(p_features)
        logging.info(f'returning new state and reward {reward}')
        return np.array(p_features), reward, self.terminated, self.truncated, self.info

    def reset(self, seed=1, options=None):
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


class TeamRunner:
    agent_type = 'state'
    _error = None
    _total_act_time = 0

    def __init__(self, team_or_dir):
        from pathlib import Path
        try:
            from grader import grader
        except ImportError:
            try:
                from . import grader
            except ImportError:
                import grader

        self._error = None
        self._team = None
        try:
            if isinstance(team_or_dir, (str, Path)):
                assignment = grader.load_assignment(team_or_dir)
                if assignment is None:
                    self._error = 'Failed to load submission.'
                else:
                    self._team = assignment.Team()
            else:
                self._team = team_or_dir
        except Exception as e:
            self._error = 'Failed to load submission: {}'.format(str(e))
        if hasattr(self, '_team') and self._team is not None:
            self.agent_type = self._team.agent_type

    def new_match(self, team: int, num_players: int) -> list:
        self._total_act_time = 0
        self._error = None
        try:
            r = self._team.new_match(team, num_players)
            if isinstance(r, str) or isinstance(r, list) or r is None:
                return r
            self._error = 'new_match needs to return kart names as a str, list, or None. Got {!r}!'.format(r)
        except Exception as e:
            self._error = 'Failed to start new_match: {}'.format(str(e))
        return []

    def act(self, player_state, *args, **kwargs):
        from time import time
        t0 = time()
        try:
            r = self._team.act(player_state, *args, **kwargs)
        except Exception as e:
            self._error = 'Failed to act: {}'.format(str(e))
        else:
            self._total_act_time += time()-t0
            return r
        return []

    def info(self):
        return RunnerInfo(self.agent_type, self._error, self._total_act_time)


class MatchException(Exception):
    def __init__(self, score, msg1, msg2):
        self.score, self.msg1, self.msg2 = score, msg1, msg2


# class CustomEnv(gym.Env):
#     """Custom Environment that follows gym interface."""

#     def __init__(self):
#         # STEP0
#         super(CustomEnv).__init__()
#         # Define action and observation space
#         # They must be gym.spaces objects
#         # Example when using discrete actions:
#         self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
#         # # Example for using image as input (channel-first; channel-last also works):
#         # self.observation_space = spaces.Box(low=0, high=255,
#         #                                     shape=(N_CHANNELS, HEIGHT, WIDTH), dtype=np.uint8)
#         self.observation_space = spaces.Dict({})

#     def step(self, action):
#         ...
#         return observation, reward, terminated, truncated, info

#     def reset(self, seed=None, options=None):
#         # STEP 1
#         ...
#         return observation, info

#     def render(self):
#         ...

#     def close(self):
#         ...

# if __name__ == '__main__':
#     from argparse import ArgumentParser
#     from pathlib import Path
#     from os import environ
#     from . import remote, utils

#     parser = ArgumentParser(description="Play some Ice Hockey. List any number of players, odd players are in team 1, even players team 2.")
#     parser.add_argument('-r', '--record_video', help="Do you want to record a video?")
#     parser.add_argument('-s', '--record_state', help="Do you want to pickle the state?")
#     parser.add_argument('-f', '--num_frames', default=1200, type=int, help="How many steps should we play for?")
#     parser.add_argument('-p', '--num_players', default=2, type=int, help="Number of players per team")
#     parser.add_argument('-m', '--max_score', default=3, type=int, help="How many goal should we play to?")
#     parser.add_argument('-j', '--parallel', type=int, help="How many parallel process to use?")
#     parser.add_argument('--ball_location', default=[0, 0], type=float, nargs=2, help="Initial xy location of ball")
#     parser.add_argument('--ball_velocity', default=[0, 0], type=float, nargs=2, help="Initial xy velocity of ball")
#     parser.add_argument('team1', help="Python module name or `AI` for AI players.")
#     parser.add_argument('team2', help="Python module name or `AI` for AI players.")
#     args = parser.parse_args()

#     logging.basicConfig(level=environ.get('LOGLEVEL', 'WARNING').upper())



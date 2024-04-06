import gymnasium
import numpy as np
import pystk
from gymnasium import spaces

import logging
import numpy as np
from collections import namedtuple
from utils import VideoRecorder

TRACK_NAME = 'icy_soccer_field'
MAX_FRAMES = 1000

def limit_period(angle):
    # turn angle into -1 to 1
    return angle - torch.floor(angle / 2 + 0.5) * 2

def extract_state_train(p_states, opponent_state, soccer_state, team_id):
    # features of ego-vehicle
    kart_front = torch.tensor(p_states['kart']['front'], dtype=torch.float32)[[0, 2]]
    kart_center = torch.tensor(p_states['kart']['location'], dtype=torch.float32)[[0, 2]]
    kart_direction = (kart_front-kart_center) / torch.norm(kart_front-kart_center)
    kart_angle = torch.atan2(kart_direction[1], kart_direction[0])

    # features of soccer
    puck_center = torch.tensor(soccer_state['ball']['location'], dtype=torch.float32)[[0, 2]]
    kart_to_puck_direction = (puck_center - kart_center) / torch.norm(puck_center-kart_center)
    kart_to_puck_angle = torch.atan2(kart_to_puck_direction[1], kart_to_puck_direction[0])

    kart_to_puck_angle_difference = limit_period((kart_angle - kart_to_puck_angle)/np.pi)

    # features of opponents
    opponent_center0 = torch.tensor(opponent_state[0]['kart']['location'], dtype=torch.float32)[[0, 2]] if len(opponent_state) else torch.tensor((10,0), dtype=torch.float32)
    opponent_center1 = torch.tensor(opponent_state[1]['kart']['location'], dtype=torch.float32)[[0, 2]] if len(opponent_state) else torch.tensor((10,0), dtype=torch.float32)

    kart_to_opponent0 = (opponent_center0 - kart_center) / torch.norm(opponent_center0-kart_center)
    kart_to_opponent1 = (opponent_center1 - kart_center) / torch.norm(opponent_center1-kart_center)

    kart_to_opponent0_angle = torch.atan2(kart_to_opponent0[1], kart_to_opponent0[0])
    kart_to_opponent1_angle = torch.atan2(kart_to_opponent1[1], kart_to_opponent1[0])

    kart_to_opponent0_angle_difference = limit_period((kart_angle - kart_to_opponent0_angle)/np.pi)
    kart_to_opponent1_angle_difference = limit_period((kart_angle - kart_to_opponent1_angle)/np.pi)

    # features of score-line
    goal_line_center = torch.tensor(soccer_state['goal_line'][team_id], dtype=torch.float32)[:, [0, 2]].mean(dim=0)

    puck_to_goal_line = (goal_line_center-puck_center) / torch.norm(goal_line_center-puck_center)
    puck_to_goal_line_angle = torch.atan2(puck_to_goal_line[1], puck_to_goal_line[0])
    kart_to_goal_line_angle_difference = limit_period((kart_angle - puck_to_goal_line_angle)/np.pi)

    features = torch.tensor([kart_center[0], kart_center[1], kart_angle, kart_to_puck_angle, opponent_center0[0],
        opponent_center0[1], opponent_center1[0], opponent_center1[1], kart_to_opponent0_angle, kart_to_opponent1_angle,
        goal_line_center[0], goal_line_center[1], puck_to_goal_line_angle, kart_to_puck_angle_difference,
        kart_to_opponent0_angle_difference, kart_to_opponent1_angle_difference,
        kart_to_goal_line_angle_difference], dtype=torch.float32)
    return features

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

class DummyTeam():
    def __init__(self, num_players, team_id):
        self.num_players = num_players
        self.team_id = team_id
    def new_match(self):
        return ['tux'] * self.num_players
    def act(self, action):
        # return [dict(acceleration=action[0], steer=action[1], brake=True if action[2] > 0.05 else False, nitro=True if action[3] > 0.05 else False, drift=True if action[4] > 0.05 else False, rescue=False, fire=False)]
        return [dict(acceleration=action[0], steer=action[1], brake=False, nitro=False, drift=False, rescue=False, fire=False)]
        # return [dict(acceleration=action[0]/10, steer=action[1]/100, brake=True if action[2] >0 else False, nitro=True if action[3] >0 else False, drift=True if action[4] >0 else False, rescue=False, fire=False)]
    def reset(self):
        pass

class IceHockeyEnvImitation(gymnasium.Env):
    def __init__(self, args, logging_level=None):
        self.num_envs =1
        self.do_init = True
        self.args = args
        self.logging_level = logging_level
        pi = 3.2
        super(IceHockeyEnvImitation, self).__init__()
        #
        self.action_space = spaces.Box(low=np.array([0, -1]), high=np.array([1, 1]), dtype=np.float32)
        # kart_center[0] - 0 to 100 p1_x
        # kart_center[1] - 0 to 100 p2_y
        # kart_angle - -pi to pi
        # kart_to_puck_angle - -pi to pi
        # opponent_center0[0] - 0 to 100(should be less) p1_x
        # opponent_center0[1] - 0 to 100(should be less) p1_y
        # opponent_center1[0] - 0 to 100(should be less) p2_x
        # opponent_center1[1] - 0 to 100(should  be less) p2_y
        # goal_line_center - our goal x
        # goal_line_center - our  goal y
        # puck_to_goal_line_angle - -pi to pi
        # kart_to_puck_angle_difference - - 1 to 1

        # kart_to_opponent0_angle - -pi to pi
        # kart_to_opponent1_angle - -pi to pi
        # kart_to_opponent0_angle_difference - -1 to 1
        # kart_to_opponent1_angle_difference - -1 to 1
        # kart_to_goal_line_angle_difference
        self.observation_space = spaces.Box(low=np.array([0, 0, -pi, -pi, 0, 0, 0, 0, 0, 0, -pi, -1, -pi, -pi, -1, -1, -1]), high=np.array([100, 100, pi, pi, 100, 100, 100, 100, 60, 60, -pi, 1, pi, pi, 1, 1, 1]),dtype=np.float32)
        # features = torch.tensor([kart_center[0], kart_center[1], kart_angle, kart_to_puck_angle, opponent_center0[0],
        #                          opponent_center0[1], opponent_center1[0], opponent_center1[1], kart_to_opponent0_angle,
        #                          kart_to_opponent1_angle,
        #                          goal_line_center[0], goal_line_center[1], puck_to_goal_line_angle,
        #                          kart_to_puck_angle_difference,
        #                          kart_to_opponent0_angle_difference, kart_to_opponent1_angle_difference,
        #                          kart_to_goal_line_angle_difference], dtype=torch.float32)

        self._pystk = pystk
        self._pystk.init(self._pystk.GraphicsConfig.none())
        if self.logging_level is not None:
            logging.basicConfig(level=self.logging_level)
        self.recorder = VideoRecorder(self.args.record_fn) if self.args.record_fn else None

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
        p_features = extract_state_train(team1_state_next, team2_state_next, soccer_state, 0).flatten().tolist()

        # reward = self.reward.step(p_features)
        # logging.info(f'returning new state and reward {reward}')
        # print(f"reward: {reward}")
        return np.array(p_features), 0, self.terminated and self.truncated, self.info

    def reset(self, seed=1, options=None):
        # super().reset(seed=seed)
        logging.info('Resetting')
        # self.recorder = VideoRecorder('infer.mp4') if self.args.record_fn else None
        # self.reward = Reward()
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
        p_features = extract_state_train(team1_state_next[0], team2_state_next, soccer_state, 0).flatten().tolist()
        # print(p_features)
        return np.array(p_features)

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
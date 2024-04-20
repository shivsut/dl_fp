import gymnasium, pystk, logging
import numpy as np
from gymnasium import spaces
import numpy as np
from collections import namedtuple
from imitation_agent.utils import VideoRecorder, discretization
from imitation_agent.features import extract_features, extract_featuresV2


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

class DummyTeam():
    def __init__(self, num_players, team_id):
        self.num_players = num_players
        self.team_id = team_id
    def new_match(self):
        return ['tux'] * self.num_players
    def act(self, action):
        # return [dict(acceleration=action[0], steer=action[1], brake=True if action[2] > 0.05 else False, nitro=True if action[3] > 0.05 else False, drift=True if action[4] > 0.05 else False, rescue=False, fire=False)]
        return [dict(acceleration=action[0], steer=action[1], brake=action[2]>0.5, nitro=False, drift=False, rescue=False, fire=False)]
        # return [dict(acceleration=action[0]/10, steer=action[1]/100, brake=True if action[2] >0 else False, nitro=True if action[3] >0 else False, drift=True if action[4] >0 else False, rescue=False, fire=False)]
    def reset(self):
        pass

class IceHockeyLearner(gymnasium.Env):
    def __init__(self, args, expert='jurgen_agent', logging_level=None, print_episode_result=False):
        self.num_envs =1
        self.do_init = True
        self.args = args
        self.logging_level = logging_level
        self.print_episode_result = print_episode_result
        pi = np.pi
        x_max = 47
        y_max = 80
        goal_center = 64.5 if args.team == "blue" else -64.5
        # pi = 3.2
        self.extract_state_train = extract_featuresV2 if expert=='jurgen_agent' else extract_features
        super(IceHockeyLearner, self).__init__()
        if args.md:
            self.discrete = discretization(aceel_div=args.md)
            self.action_space = spaces.MultiDiscrete(nvec=self.discrete.K_n, start=self.discrete.start)
        else:
            self.discrete = None
            self.action_space = spaces.Box(low=np.array([0, -1, 0]), high=np.array([1, 1, 1]), dtype=np.float32)
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
        if expert == 'jurgen_agent':
            # TODO optimize max space
            # kart_center[0],
            # kart_center[1],
            # kart_angle,
            # kart_to_puck_angle,
            #                          goal_line_center[0],
            #                          goal_line_center[1],
            #                          kart_to_puck_angle_difference,
            #                          puck_center[0],
            #                          puck_center[1],
            #                          puck_to_goal_line[0],
            #                          puck_to_goal_line[1]],
            #                         dtype=torch.float32)
            self.observation_space = spaces.Box(low=np.array([-x_max, -y_max, -pi, -pi, -0.1, goal_center-1, -1, -x_max,  -y_max,  -1,  0]),
                                                high=np.array([x_max,  y_max,  pi,  pi,  0.1, goal_center+1,  1,  x_max,   y_max,   1,  1]),
                                                dtype=np.float32)
        else:
            self.observation_space = spaces.Box(low=np.array([-x_max, -y_max, -pi, -pi, -x_max, -y_max, -x_max, -y_max, -pi, -pi, -0.1, goal_center-1, -pi, -1, -1, -1, -1]),
                                                high=np.array([x_max,  y_max,  pi,  pi,  x_max,  y_max,  x_max,  y_max,  pi,  pi,  0.1, goal_center+1,  pi,  1,  1,  1, 1]),
                                                dtype=np.float32)
        # features = torch.tensor([kart_center[0],
        # kart_center[1],
        # kart_angle,
        # kart_to_puck_angle,
        # opponent_center0[0],
        #
        #                          opponent_center0[1],
        #                          opponent_center1[0],
        #                          opponent_center1[1],
        #                          kart_to_opponent0_angle,
        #                          kart_to_opponent1_angle,
        #                          goal_line_center[0],
        #                          goal_line_center[1],
        #                          puck_to_goal_line_angle,
        #                          kart_to_puck_angle_difference,
        #                          kart_to_opponent0_angle_difference,
        #                          kart_to_opponent1_angle_difference,
        #                          kart_to_goal_line_angle_difference], dtype=torch.float32)

        self._pystk = pystk
        self._pystk.init(self._pystk.GraphicsConfig.none())
        if self.logging_level is not None:
            logging.basicConfig(level=self.logging_level)
        self.recorder = VideoRecorder(self.args.record_fn) if self.args.record_fn else None

        self.timeout = 1e10
        self.max_timestep = 500
        self.current_timestep = 0
        self.max_score = 1
        self.num_players = 1
        self.my_team_id = 0
        self.opp_team_id = 1
        if args.team=="red":
            self.my_team_id, self.opp_team_id = self.opp_team_id, self.my_team_id
        self.my_team = DummyTeam(self.num_players, self.my_team_id)
        if self.args.use_opponent:
            self.opponent_team = AIRunner() if self.args.opponent == 'ai' else TeamRunner(args.opponent)

        self.info = {}
        num_kart = 2 if self.args.use_opponent else 1
        self.race_config = self._pystk.RaceConfig(track=TRACK_NAME, mode=self._pystk.RaceConfig.RaceMode.SOCCER, num_kart=num_kart * self.num_players)
        self.race_config.players.pop()
        player_cfgs = [{"team_id":self.my_team_id, "is_ai": False, "kart":'tux'}]
        player_cfgs.append({"team_id": self.opp_team_id, "is_ai": True if args.opponent == 'ai' else False, "kart": 'tux'})
        if args.team == "red":
            player_cfgs.reverse()
        for cfg in player_cfgs:
            self.race_config.players.append(self._make_config(**cfg))


    def _make_config(self, team_id, is_ai, kart):
        PlayerConfig = self._pystk.PlayerConfig
        controller = PlayerConfig.Controller.AI_CONTROL if is_ai else PlayerConfig.Controller.PLAYER_CONTROL
        return PlayerConfig(controller=controller, team=team_id, kart=kart)

    def step(self, action):

        action = self.discrete.de_discrete(action) if self.discrete else action
        self.current_timestep += 1

        my_team_state = to_native(self.state.players[self.my_team_id])
        opp_team_state = to_native(self.state.players[self.opp_team_id])
        soccer_state = to_native(self.state.soccer)
        logging.info('calling agent')
        my_team_actions = self.my_team.act(action)
        opponent_team_actions = self.opponent_team.act([opp_team_state], [my_team_state], soccer_state) if self.args.use_opponent else []

        # TODO check for error in info and raise MatchException
        # TODO check for timeout

        if self.recorder:
            self.recorder([my_team_state], [opp_team_state], soccer_state=soccer_state, actions=my_team_actions,team1_images=None, team2_images=None)
        if self.args.opponent == 'ai':
            opponent_team_actions = [{'steer':0, 'acceleration':0, 'brake':False, 'nitro':False, 'drift':False, 'rescue':False, 'fire':False}]
        all_team_actions = my_team_actions
        if self.args.use_opponent:
            all_team_actions += opponent_team_actions
        if self.args.team == "red":
            all_team_actions.reverse()
        if (not self.race.step([self._pystk.Action(**a) for a in all_team_actions]) and self.num_players):
            self.truncated = True
        if (sum(self.state.soccer.score) >= self.max_score) or (self.current_timestep > self.max_timestep):
            self.terminated = True
        if not (self.truncated or self.terminated):
            self.state.update()
        else:
            self.num_envs = 0

        logging.info('state updated, calculating reward')
        my_team_state_next = to_native(self.state.players[self.my_team_id])
        opponent_team_state_next = to_native(self.state.players[self.opp_team_id])
        # if self.args.team =="red":
        #     team1_state_next, team2_state_next = team2_state_next, team1_state_next
        soccer_state = to_native(self.state.soccer)
        opponent_state = opponent_team_state_next
        p_features = np.array(self.extract_state_train(my_team_state_next, [opponent_state], soccer_state, self.my_team_id).flatten().tolist())
        if np.isnan(p_features).any():
            self.terminated = True

        # reward = self.reward.step(p_features)
        # logging.info(f'returning new state and reward {reward}')
        # print(f"reward: {reward}")
        # print (p_features)
        # self.terminated = True
        if self.print_episode_result and (self.truncated or self.terminated):
            print (f"Results : [{soccer_state['score'][self.my_team_id]}, {soccer_state['score'][self.opp_team_id]}] ({self.current_timestep})")
        return p_features, np.array(soccer_state['score'][self.my_team_id] - soccer_state['score'][self.opp_team_id], dtype=float), self.terminated , self.truncated, {'terminal_observation': p_features}

    def step_async(self, action):
        self.async_res= self.step(action)
    def step_wait(self):
        return self.async_res

    def reset(self, seed=1, options=None):
        self.current_timestep = 0
        self.async_res = None
        # super().reset(seed=seed)
        logging.info('Resetting')
        # self.recorder = VideoRecorder('infer.mp4') if self.args.record_fn else None
        # self.reward = Reward()
        self.truncated = False
        self.terminated = False
        logging.info('Starting new race')
        if hasattr(self, 'race'):
            self.race.stop()
            del self.race
        self.race = self._pystk.Race(self.race_config)
        self.race.start()
        if self.args.use_opponent:
            self.opponent_team.new_match(self.opp_team_id, 1)
        self.my_team.new_match()
        self.state = self._pystk.WorldState()
        self.state.update() # TODO need to call this here?

        my_team_state = to_native(self.state.players[self.my_team_id])
        opp_team_state = to_native(self.state.players[self.opp_team_id])
        # if self.args.team=="red":
        #     team1_state_next, team2_state_next = team2_state_next, team1_state_next
        soccer_state = to_native(self.state.soccer)
        opponent_state = opp_team_state
        p_features = self.extract_state_train(my_team_state, [opponent_state], soccer_state, self.my_team_id).flatten().tolist()
        return np.array(p_features), {'terminal_observation': np.array(p_features)}

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

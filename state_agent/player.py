import time
from os import path

import torch
import numpy as np

from imitation_agent.features import extract_features, extract_featuresV2



class Team:
    agent_type = 'state'
    def __init__(self):
        """
          TODO: Load your agent here. Load network parameters, and other parts of our model
          We will call this function with default arguments only
        """
        self.team = None
        self.num_players = None
        self.verbose = False
        self.use_model = False
        self.device = "cpu" # torch.device("cuda" if torch.cuda.is_available() else "cpu")

        team_blue = "/dl_fp_main/AI_L2x256_blue/AI_L2x256_blue_jit.pt"
        team_red = "/dl_fp_main/AI_L2x256_red/AI_L2x256_red_jit.pt"
        self.model_blue = torch.jit.load(team_blue, map_location='cpu')
        self.model_red = torch.jit.load(team_red, map_location='cpu')
        self.model = self.model_blue
        self.running_avg = []

    def new_match(self, team: int, num_players: int) -> list:
        """
        Let's start a new match. You're playing on a `team` with `num_players` and have the option of choosing your kart
        type (name) for each player.
        :param team: What team are you playing on RED=0 or BLUE=1
        :param num_players: How many players are there on your team
        :return: A list of kart names. Choose from 'adiumy', 'amanda', 'beastie', 'emule', 'gavroche', 'gnu', 'hexley',
                 'kiki', 'konqi', 'nolok', 'pidgin', 'puffy', 'sara_the_racer', 'sara_the_wizard', 'suzanne', 'tux',
                 'wilber', 'xue'. Default: 'tux'
        """
        """
           TODO: feel free to edit or delete any of the code below
        """
        if team==1:
            self.model = self.model_red
        self.team, self.num_players = team, num_players
        return ['tux'] * num_players


    def act(self, player_state, opponent_state, soccer_state):
        """
        This function is called once per timestep. You're given a list of player_states and images.

        DO NOT CALL any pystk functions here. It will crash your program on your grader.

        :param player_state: list[dict] describing the state of the players of this team. The state closely follows
                             the pystk.Player object <https://pystk.readthedocs.io/en/latest/state.html#pystk.Player>.
                             You can ignore the camera here.
                             kart:  Information about the kart itself
                               - front:     float3 vector pointing to the front of the kart
                               - location:  float3 location of the kart
                               - rotation:  float4 (quaternion) describing the orientation of kart (use front instead)
                               - size:      float3 dimensions of the kart
                               - velocity:  float3 velocity of the kart in 3D

        :param opponent_state: same as player_state just for other team

        :param soccer_state: dict  Mostly used to obtain the puck location
                             ball:  Puck information
                               - location: float3 world location of the puck

        :return: dict  The action to be taken as a dictionary. For example `dict(acceleration=1, steer=0.25)`.
                 acceleration: float 0..1
                 brake:        bool Brake will reverse if you do not accelerate (good for backing up)
                 drift:        bool (optional. unless you want to turn faster)
                 fire:         bool (optional. you can hit the puck with a projectile)
                 nitro:        bool (optional)
                 rescue:       bool (optional. no clue where you will end up though.)
                 steer:        float -1..1 steering angle
        """

        actions = []
        start = time.time()
        for player_id, pstate in enumerate(player_state):
            # TODO: Use Policy to get the actions of each player
            if self.use_model:
                features = extract_featuresV2(pstate, opponent_state, soccer_state, self.team)
                features = torch.tensor(features) #.to(self.device)
                # if player_id % 2 == 0:
                #     action = self.model_p0(features)
                # else:
                #     action = self.model_p1(features)
                action = self.model(features)
                action = dict(acceleration=action[0], steer=action[1], brake=action[2])
            else:
                # Generate random actions for all players
                action = dict(acceleration=0.7, brake=0, drift=0,
                              fire=0, nitro=0, rescue=0, steer=0.3)
            
            # accumulate the action of each player
            actions.append(action)
        end = time.time()
        self.running_avg.append(end-start)
        # if len(self.running_avg) > 20:
        #     print(f"avg act time {np.mean(self.running_avg)*1000} ms")
        #     self.running_avg = []
        return actions


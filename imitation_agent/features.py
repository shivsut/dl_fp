import torch
import numpy as np 

def limit_period(angle):
    # turn angle into -1 to 1 
    return angle - torch.floor(angle / 2 + 0.5) * 2 

def extract_features(pstate, opponent_state, soccer_state, team_id):
    """
    This works for all the state based agent except jurgen_agent
    """
    # features of ego-vehicle
    # import pdb; pdb.set_trace()
    kart_front = torch.tensor(pstate['kart']['front'], dtype=torch.float32)[[0, 2]]
    kart_center = torch.tensor(pstate['kart']['location'], dtype=torch.float32)[[0, 2]]
    kart_direction = (kart_front-kart_center) / torch.norm(kart_front-kart_center)
    kart_angle = torch.atan2(kart_direction[1], kart_direction[0])

    # features of soccer 
    puck_center = torch.tensor(soccer_state['ball']['location'], dtype=torch.float32)[[0, 2]]
    kart_to_puck_direction = (puck_center - kart_center) / torch.norm(puck_center-kart_center)
    kart_to_puck_angle = torch.atan2(kart_to_puck_direction[1], kart_to_puck_direction[0]) 

    kart_to_puck_angle_difference = limit_period((kart_angle - kart_to_puck_angle)/np.pi)

    # TODO: Dummy values for opponent states
    # opponent_center0 = torch.tensor([0, 0], dtype=torch.float32)
    # opponent_center1 = torch.tensor([0, 0], dtype=torch.float32)
    # opponent_center0 = torch.tensor(opponent_state[0]['kart']['location'], dtype=torch.float32)[[0, 2]]
    # opponent_center1 = torch.tensor(opponent_state[1]['kart']['location'], dtype=torch.float32)[[0, 2]]
    opponent_center0 = torch.tensor(opponent_state['kart']['location'], dtype=torch.float32)[[0, 2]]  if len(opponent_state) else torch.tensor((10,0), dtype=torch.float32)
    opponent_center1 = torch.tensor((10,0), dtype=torch.float32)

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

def extract_featuresV2(pstate, opponent_state, soccer_state, team_id):
    """
    Feature extractor function for jurgen_agent
    """
    # features of ego-vehicle
    kart_front = torch.tensor(pstate['kart']['front'], dtype=torch.float32)[[0, 2]]
    kart_center = torch.tensor(pstate['kart']['location'], dtype=torch.float32)[[0, 2]]
    kart_direction = (kart_front-kart_center) / torch.norm(kart_front-kart_center)
    kart_angle = torch.atan2(kart_direction[1], kart_direction[0])

    # features of soccer 
    puck_center = torch.tensor(soccer_state['ball']['location'], dtype=torch.float32)[[0, 2]]
    kart_to_puck_direction = (puck_center - kart_center) / torch.norm(puck_center-kart_center)
    kart_to_puck_angle = torch.atan2(kart_to_puck_direction[1], kart_to_puck_direction[0]) 

    kart_to_puck_angle_difference = limit_period((kart_angle - kart_to_puck_angle)/np.pi)

    # features of score-line 
    goal_line_center = torch.tensor(soccer_state['goal_line'][(team_id+1)%2], dtype=torch.float32)[:, [0, 2]].mean(dim=0)

    puck_to_goal_line = (goal_line_center-puck_center) / torch.norm(goal_line_center-puck_center)

    features = torch.tensor([kart_center[0], kart_center[1], kart_angle, kart_to_puck_angle, 
        goal_line_center[0], goal_line_center[1], kart_to_puck_angle_difference, 
        puck_center[0], puck_center[1], puck_to_goal_line[0], puck_to_goal_line[1]], dtype=torch.float32)

    return features 
import torch
import numpy as np
import torch.nn.functional as F

def cosine_similarity(a, b):
    return np.dot(a / np.linalg.norm(a), b / np.linalg.norm(b))

def limit_period(angle):
    # turn angle into -1 to 1
    return angle - torch.floor(angle / 2 + 0.5) * 2

def extract_state_train_players(p1_state, soccer_state, opponent_state, team_id, player_type='offense'):
    # Features of vehicle 
    kart_front = torch.tensor(p1_state['kart']['front'], dtype=torch.float32)[[0, 2]]
    kart_center = torch.tensor(p1_state['kart']['location'], dtype=torch.float32)[[0, 2]]
    kart_direction = (kart_front-kart_center) / torch.norm(kart_front-kart_center)
    kart_angle = torch.atan2(kart_direction[1], kart_direction[0])

    # features of soccer
    puck_center = torch.tensor(soccer_state['ball']['location'], dtype=torch.float32)[[0, 2]]
    kart_to_puck_direction = (puck_center - kart_center) / torch.norm(puck_center - kart_center)
    kart_to_puck_angle = torch.atan2(kart_to_puck_direction[1], kart_to_puck_direction[0])
    kart_to_puck_angle_difference = limit_period((kart_angle - kart_to_puck_angle) / np.pi)

    # Vehicle - Soccer features
    kart_to_puck_dist = torch.norm(puck_center - kart_front)

    # features of score-line
    goal_line_center = torch.tensor(soccer_state['goal_line'][team_id], dtype=torch.float32)[:, [0, 2]].mean(dim=0)
    puck_to_goal_line = (goal_line_center - puck_center) / torch.norm(goal_line_center - puck_center)
    puck_to_goal_line_angle = torch.atan2(puck_to_goal_line[1], puck_to_goal_line[0])
    kart_to_goal_line_angle_difference = limit_period((kart_angle - puck_to_goal_line_angle) / np.pi)

    # Align "player-puck-opponent goal post"
    v1 = (puck_center-kart_center)
    v2 = (goal_line_center-kart_center)
    alignment = cosine_similarity(v1, v2)

    # Goal distance 
    goal_dist = torch.norm(goal_line_center - puck_center)
 
    # Goal and puck distance 
    puck_and_goal_distance = torch.norm(puck_center - goal_line_center)

    # features of score
    output = [kart_to_puck_dist, alignment, goal_dist, puck_and_goal_distance]

    return output

def extract_state_train1(p_states, opponent_state, soccer_state, team_id):
    res = []
    res.extend(extract_state_train_players(p_states[0], soccer_state, opponent_state, team_id))
    if len(p_states) > 1:
        res.extend(extract_state_train_players(p_states[1], soccer_state, opponent_state, team_id, player_type='defense'))
    return res

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
    opponent_center0 = torch.tensor(opponent_state[0]['kart']['location'], dtype=torch.float32)[[0, 2]] if len(opponent_state) else torch.tensor((0,0), dtype=torch.float32)
    opponent_center1 = torch.tensor(opponent_state[1]['kart']['location'], dtype=torch.float32)[[0, 2]] if len(opponent_state) else torch.tensor((0,0), dtype=torch.float32)

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
def extract_state_infer(p_states, soccer_state, opponent_state, team_id):
    res = []
    res.extend(extract_state_train_players(p_states[0], soccer_state, opponent_state, team_id))
    if len(p_states) > 1:
        res.extend(extract_state_train_players(p_states[1], soccer_state, opponent_state, team_id, player_type='defense'))
    return res
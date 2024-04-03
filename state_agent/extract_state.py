
import torch
def extract_state_train_p1(p1_state, soccer_state, opponent_state, team_id):
    kart_front = torch.tensor(p1_state['kart']['front'], dtype=torch.float32)[[0, 2]]
    puck_center = torch.tensor(soccer_state['ball']['location'], dtype=torch.float32)[[0, 2]]
    # kart_to_puck_dist = (puck_center - kart_front) / torch.norm(puck_center - kart_front)
    kart_to_puck_dist = torch.norm(puck_center - kart_front)
    return kart_to_puck_dist

def extract_state_train_p2(p2_state, soccer_state, opponent_state, team_id):
    kart_front = torch.tensor(p2_state['kart']['front'], dtype=torch.float32)[[0, 2]]
    puck_center = torch.tensor(soccer_state['ball']['location'], dtype=torch.float32)[[0, 2]]
    # kart_to_puck_dist = (puck_center - kart_front) / torch.norm(puck_center - kart_front)
    kart_to_puck_dist = torch.norm(puck_center - kart_front)
    return kart_to_puck_dist

def extract_state_train(p_states, opponent_state, soccer_state, team_id):
    res = []
    res.append(extract_state_train_p1(p_states[0], soccer_state, opponent_state, team_id))
    if len(p_states) > 1:
        res.append(extract_state_train_p1(p_states[1], soccer_state, opponent_state, team_id))
    return res

def extract_state_infer(p_states, soccer_state, opponent_state, team_id):
    res = []
    res.append(extract_state_train_p1(p_states[0], soccer_state, opponent_state, team_id))
    res.append(extract_state_train_p1(p_states[1], soccer_state, opponent_state, team_id))
    return res

class Reward:
    def __init__(self):
        self.prev_reward = 0
    def step(self, p_states):
        curr_reward = -p_states[0]
        if len(p_states) > 1:
            curr_reward += -p_states[1]
        curr_reward *= 0.5
        curr_reward += curr_reward - self.prev_reward
        self.prev_reward = curr_reward
        return curr_reward
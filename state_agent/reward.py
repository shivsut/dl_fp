
class Reward:
    def __init__(self):
        self.prev_reward = 0
        self.not_first = False
    def step(self, p_states):
        curr_reward = -p_states[0]
        if len(p_states) > 1:
            curr_reward += -p_states[1]
        curr_reward *= 0.5
        if self.not_first:
            curr_reward += (curr_reward - self.prev_reward)
        self.not_first = True
        self.prev_reward = curr_reward
        return curr_reward
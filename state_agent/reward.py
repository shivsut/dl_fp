
class Reward:
    def __init__(self):
        self.prev_reward = 0
    def step(self, p_states):
        curr_reward = (p_states[0] + p_states[1])
        self.prev_reward = curr_reward
        return -curr_reward
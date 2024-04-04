import numpy as np 


class Reward:

    def __init__(self):
        self.prev_reward = 0
        self.not_first = False

    def step(self, p_states):
        # print(f"p_state: {p_states[0]}")
        # import pdb; pdb.set_trace()
        curr_reward = 100*np.exp(-p_states[0])
        if len(p_states) > 1:
            curr_reward += -p_states[1]
        # curr_reward *= 0.01
        if self.not_first:
            curr_reward += (curr_reward - self.prev_reward)
        self.not_first = True
        self.prev_reward = curr_reward
        return curr_reward
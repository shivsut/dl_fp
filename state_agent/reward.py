import numpy as np 


class Reward:

    def __init__(self):
        self.prev_reward = 0
        self.not_first = False
    
    def funcV1(self, p_states):
        """
        Version1 of calculating the reward based on only the 
        distance b/w the player and puck.
        """
        # Function to calculate the reward given the function states
        rewardStates = lambda x: 100 * np.exp(-x)

        # Calculate reward for player 1 
        curr_reward = rewardStates(p_states[0])
        
        # If 2nd players is in game
        if len(p_states) > 1:
            # Calculate reward for player 2
            curr_reward += -p_states[1]
        
        # Accumulate the reward
        if self.not_first:
            curr_reward += (curr_reward - self.prev_reward)
        self.not_first = True
        self.prev_reward = curr_reward
        return curr_reward

    def rewardStatesV2(self, p_states):
        # [kart_to_puck_dist, alignment, goal_dist, puck_and_goal_distance]
        reward = 0
        # Minimize distance b/w cart and puck
        # import pdb; pdb.set_trace()
        reward += 100 * np.exp(-p_states[0])
        # Maximize alignment  
        reward += 0.001 * (-p_states[1])

        return reward

    def funcV2(self, p_states):
        """
        Version2 of calculating the reward based on
        1) the distance b/w the player and puck and, 
        2) alignment
        """
        # Calculate reward for player 1 
        curr_reward = self.rewardStatesV2(p_states)
        
        # If 2nd players is in game
        if len(p_states) > 1:
            # Calculate reward for player 2
            curr_reward += -p_states[1]
        
        # Accumulate the reward
        if self.not_first:
            curr_reward += (curr_reward - self.prev_reward)
        self.not_first = True
        self.prev_reward = curr_reward
        return curr_reward

    def funcV3(self, p_states):
        """
        Version3 of calculating the reward based on
        1) the distance b/w the player and puck and, 
        2) kart angle
        3) Player1: Offense and, Player2: Defense
        """
    
    def step(self, p_states, version='V2'):
        if version=='V1':
            return self.funcV1(p_states)
        elif version=='V2':
            return self.funcV2(p_states)
        
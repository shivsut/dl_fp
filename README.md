# Deep Learning Spring 2024 - Final Project

## Commands for creating submission
```bash
python3 -m grader state_agent -v
python3 bundle.py state_agent group33
python3 -m grader group33.zip -v
```
## Demo for creating Canvas compatible jit file
Training Command:
```bash
python3 -m imitation_agent.train -e 1 -v AI_L2x256_blue --time_steps 500000 --time_steps_infer 10 --nenv 2 --use_opponent --expert jurgen_agent --batch_size 512 --device cuda --md 90 --net_arch "256,256"
```
Command for creating jit compatible file:
```bash
python3 -m imitation_agent.canvas_jit -e 1 -v AI_L2x256_blue_error --time_steps 1 --time_steps_infer 10 --nenv 1 --use_opponent --expert jurgen_agent --batch_size 512 --device cuda --md 90 --net_arch "256,256" --resume_training "AI_L2x256_blue/AI_L2x256_blue.pt"
```

## Reward Function for single team match

### Offense Player (Player 1)
1) Minimize "kart_to_puck_dist"
    - Rewarding the player (it increases exponentially): np.exp(-x)
    - TODO: Check if normalizing the values helps or not
2) Aligning the player-puck-opponent goal post
    - 1st vector: (puck - player)
    - 2nd vector: (opponent goal post - player)
    - Use cosine similarity b/w the two vectors
3) Reward for scoring the goal
    - Minimize the distance b/w ball and goal post

### Defense Player (Player 2)

1) Reward for not allowing the ball in a region

## Reward Function for match against opponents

1) Reward based on current match state
2) 

***
## Notes

### PyStk State space

**player_state**
 - camera (https://pystk.readthedocs.io/en/latest/state.html#pystk.Camera) 
 - Not needed????

 - Kart (https://pystk.readthedocs.io/en/latest/state.html#pystk.Kart)
 - attachment - types of attachment
 - front - Front direction of kart 1/2 kart length forward from location - float3
 - id - Kart id compatible with instance labels - int
 - jumping - Is the kart jumping? - bool (Not needed I think)
 -  location - 3D world location of the kart - float3
 - max_steer_angle - Maximum steering angle - float
 - name - Player name - str
 - overall_distance - Overall distance traveled - float (Not needed I think)
 - player_id - Player id - int
 - powerup - Powerup collected - powerup
 - rotation - Quaternion rotation of the kart - Quaternion
 - velocity - Velocity of kart - float3

**game_state**
- ball
- id - Object id of the soccer ball - int
- location - 3D world location of the item - float3 ( )
- size - Size of the ball - float

- goal_line (static) -  Start and end of the goal line for each team - List[List[float3[2]][2]]


### Features calculated by extract_features()
- kart_direction - float2
- kart_angle - float
- kart_to_puck_direction - float2
- kart_to_puck_angle - float
- kart_to_puck_angle_difference - float

- kart_to_opponent0 - float2
- kart_to_opponent0_angle -  float
- kart_to_opponent0_angle_difference - float

- goal_line_center  - float2

- puck_to_goal_line - float2
- puck_to_goal_line_angle - float
- kart_to_goal_line_angle_difference - float

Somethings that can help
- ball velocity
- ball acceleration?

### Rewards (https://github.com/Rolv-Arild/Necto)
- agents velocity towards ball
- balls velocity towards goal
- +ve reward on goal
+ -ve reward on opponent goal
+ +ve reward if shot on target
+ +ve on making a save
+ +ve on impeding opponent


psuedo code for necto rewards
- game state reward = ball_pos closer to goal (continuos)


### Questions
- How does it know to reverse?
- continous space
- how can i train with some base agent
- circiculum learning - ppo 
- action space 
- rewarding shaping

***
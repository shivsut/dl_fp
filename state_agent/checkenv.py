from stable_baselines3.common.env_checker import check_env
from env import IceHockeyEnv


env = IceHockeyEnv()
# It will check your custom environment and output additional warnings if needed
check_env(env)

# Training loop
# episodes = 500
# for episode in range(episodes):
# 	done = False
# 	env.reset()
# 	# while True:#not done:
# 	for _ in range(2):#not done:
# 		random_action = env.action_space.sample()
# 		print("action",random_action)
# 		observation, reward, terminated, truncated, info = env.step(random_action)
# 		print('reward',reward)
# 		if terminated or truncated:
# 			observation, info = env.reset()

# env.close()
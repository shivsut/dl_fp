from argparse import ArgumentParser

import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from .env import IceHockeyEnv

# https://stable-baselines3.readthedocs.io/en/master/guide/examples.html

# Parallel environments
vec_env = make_vec_env("CartPole-v1", n_envs=4)

parser = ArgumentParser()
parser.add_argument('-r', '--record_fn', default=None)
parser.add_argument('--do_train', action='store_true')
parser.add_argument('-t', '--time_steps', type=int, default=50000)
parser.add_argument('-ti', '--time_steps_infer', type=int, default=1000)
parser.add_argument('-m', '--model_name', default='hockey')
parser.add_argument('-d', '--deterministic', action='store_true')

args = parser.parse_args()


vec_env = IceHockeyEnv(args, logging_level='ERROR')

if args.do_train:
    # vec_env = IceHockeyEnv(args, logging_level='ERROR')
    model = PPO("MlpPolicy", vec_env, verbose=1)
    model.learn(total_timesteps=args.time_steps)
    model.save(args.model_name)

else:
    model = PPO.load(args.model_name, env=vec_env)
    # Enjoy trained agent
    vec_env = model.get_env()
    obs = vec_env.reset()
    for i in range(args.time_steps_infer):
        action, _states = model.predict(obs, deterministic=args.deterministic)
        obs, rewards, dones, info = vec_env.step(action)
        print(rewards, end="\t")
from argparse import ArgumentParser

import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv

from .env import IceHockeyEnv

# https://stable-baselines3.readthedocs.io/en/master/guide/examples.html
if __name__ == '__main__':
    # Parallel environments
    # vec_env = make_vec_env("CartPole-v1", n_envs=4)
    parser = ArgumentParser()
    parser.add_argument('-n', '--nenv', type=int, default=1)
    parser.add_argument('-r', '--record_fn', default=None)
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('-t', '--time_steps', type=int, default=100000)
    parser.add_argument('-ti', '--time_steps_infer', type=int, default=1000)
    parser.add_argument('-m', '--model_name', default='hockey')
    parser.add_argument('-d', '--deterministic', action='store_true')
    parser.add_argument( '--tensor_log', action='store_true')
    parser.add_argument( '--debug_mode', action='store_true')

    args = parser.parse_args()
    if args.do_train:
        if not args.debug_mode:
            # pdb debugger doesn't work with multiple processes
            from stable_baselines3.common.monitor import Monitor
            envs = SubprocVecEnv([lambda : Monitor(IceHockeyEnv(args, logging_level='ERROR')) for _ in range(args.nenv)])
        else:
            envs = IceHockeyEnv(args, logging_level='ERROR')
        model = PPO("MlpPolicy", envs, verbose=1, tensorboard_log='./log/' if args.tensor_log else None)
        model.learn(total_timesteps=args.time_steps)
        model.save(args.model_name)
    else:
        vec_env = IceHockeyEnv(args, logging_level='ERROR')
        model = PPO.load(args.model_name, env=vec_env)
        # Enjoy trained agent
        vec_env = model.get_env()
        obs = vec_env.reset()
        for i in range(args.time_steps_infer):
            action, _states = model.predict(obs, deterministic=args.deterministic)
            obs, rewards, dones, info = vec_env.step(action)
            # print(rewards, end="\t")
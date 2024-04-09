import os
import tempfile
from argparse import ArgumentParser

import numpy as np
import gymnasium as gym
from imitation.util.logger import HierarchicalLogger
from stable_baselines3.common.evaluation import evaluate_policy
import imitation.data.rollout as rollout

from imitation.algorithms import bc
from imitation.algorithms.dagger import SimpleDAggerTrainer
from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env
from stable_baselines3.common.logger import Logger, CSVOutputFormat
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from imitation_agent.learner import IceHockeyLearner
from imitation_agent.policy import IceHockeyEnv
from imitation_agent.utils import load_policy

# TODO: Add jurgen agent: 'jurgen_agent'
# EXPERT = ['jurgen_agent']
EXPERT = ['geoffrey_agent0', 'geoffrey_agent1', 'yann_agent', 'yoshua_agent0', 'yoshua_agent1']

def main(args):

    rng = np.random.default_rng(0)
    
    # create environment
    envs = SubprocVecEnv([lambda: Monitor(IceHockeyLearner(args, logging_level='ERROR')) for _ in range(args.nenv)])
    # BC trainer
    bc_trainer = bc.BC(
        observation_space=envs.observation_space,
        action_space=envs.action_space,
        custom_logger=HierarchicalLogger(Logger('./log/', output_formats=[CSVOutputFormat('out2.csv')])),
        rng=rng,
    )

    for expert_name in EXPERT:
        # TODO: Use the already trained checkpoint
        # TODO: 'jurgen_agent' requires different environment (observation_space is small)
        # envs = SubprocVecEnv([lambda: Monitor(IceHockeyLearner(args, expert=expert_name, logging_level='ERROR')) for _ in range(args.nenv)])
        expert = IceHockeyEnv(envs.observation_space, envs.action_space, expert_name)
        with tempfile.TemporaryDirectory(prefix="dagger_example_") as tmpdir:
            print(tmpdir)
            dagger_trainer = SimpleDAggerTrainer(
                venv=envs,
                scratch_dir=tmpdir,
                expert_policy=expert,
                rng=rng,
                bc_trainer=bc_trainer,
                custom_logger=HierarchicalLogger(Logger('./log/', output_formats=[CSVOutputFormat(f'out_{expert_name}.csv')])),
            )
            dagger_trainer = load_policy(dagger_trainer)
            print ('training')
            dagger_trainer.train(int(args.time_steps/len(EXPERT)),
                                rollout_round_min_timesteps=0,
                                rollout_round_min_episodes=1
                                )
            dagger_trainer.policy.save("/tmp/policy/hockey.pt")
        # dagger_trainer.policy.save("./saved_model/imitation.pt")
        

    args.record_fn='opponents.mp4'
    envs_eval = SubprocVecEnv([lambda: Monitor(IceHockeyLearner(args, logging_level='ERROR')) for _ in range(1)])
    reward, _ = evaluate_policy(dagger_trainer.policy, envs_eval, args.time_steps_infer)
    print("Reward:", reward)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-n', '--nenv', type=int, default=1)
    parser.add_argument('-r', '--record_fn', default=None)
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('-t', '--time_steps', type=int, default=200)
    parser.add_argument('-ti', '--time_steps_infer', type=int, default=50)
    parser.add_argument('-m', '--model_name', default='hockey')
    parser.add_argument('-d', '--deterministic', action='store_true')
    parser.add_argument('--tensor_log', action='store_true')
    parser.add_argument('--debug_mode', action='store_true')
    parser.add_argument('--opponent', default='ai')

    args = parser.parse_args()
    main(args)

    
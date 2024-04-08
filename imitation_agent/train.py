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

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-n', '--nenv', type=int, default=1)
    parser.add_argument('-r', '--record_fn', default=None)
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('-t', '--time_steps', type=int, default=100000)
    parser.add_argument('-ti', '--time_steps_infer', type=int, default=1000)
    parser.add_argument('-m', '--model_name', default='hockey')
    parser.add_argument('-d', '--deterministic', action='store_true')
    parser.add_argument('--tensor_log', action='store_true')
    parser.add_argument('--debug_mode', action='store_true')

    rng = np.random.default_rng(0)
    # env = make_vec_env(
    #     "seals:seals/CartPole-v0",
    #     rng=rng,
    # )
    args = parser.parse_args()
    # env = IceHockeyEnvImitation(args, logging_level='ERROR')
    envs = SubprocVecEnv([lambda: Monitor(IceHockeyLearner(args, logging_level='ERROR')) for _ in range(args.nenv)])
    expert = IceHockeyEnv(envs.observation_space, envs.action_space)

    # expert = IceHockeyEnv()
    # expert = load_policy(
    #     "ppo-huggingface",
    #     organization="HumanCompatibleAI",
    #     env_name="seals-CartPole-v0",
    #     venv=env,
    # )

    bc_trainer = bc.BC(
        observation_space=envs.observation_space,
        action_space=envs.action_space,
        custom_logger=HierarchicalLogger(Logger('./log/', output_formats=[CSVOutputFormat('out2.csv')])),
        rng=rng,
    )
    with tempfile.TemporaryDirectory(prefix="dagger_example_") as tmpdir:
        print(tmpdir)
        dagger_trainer = SimpleDAggerTrainer(
            venv=envs,
            scratch_dir=tmpdir,
            expert_policy=expert,
            rng=rng,
            bc_trainer=bc_trainer,
            custom_logger=HierarchicalLogger(Logger('./log/', output_formats=[CSVOutputFormat('out1.csv')])),
        )
        print ('training')
        dagger_trainer.train(5000,
                             rollout_round_min_timesteps=0,
                             rollout_round_min_episodes=1
                             )
        print('training done, evaluating')
        dagger_trainer.policy.save("./saved_model/imitation.pt")

    args.record_fn='infer.mp4'
    envs_eval = SubprocVecEnv([lambda: Monitor(IceHockeyLearner(args, logging_level='ERROR')) for _ in range(1)])
    # envs_eval = envs
    reward, _ = evaluate_policy(dagger_trainer.policy, envs_eval, 2)
    print("Reward:", reward)
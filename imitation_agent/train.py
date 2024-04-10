import os
import tempfile
from argparse import ArgumentParser
from random import shuffle
import numpy as np
import gymnasium as gym
from imitation.util.logger import HierarchicalLogger
from stable_baselines3.common.evaluation import evaluate_policy
import imitation.data.rollout as rollout

from imitation.algorithms import bc
from imitation.algorithms.dagger import SimpleDAggerTrainer
from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env
from stable_baselines3.common.logger import Logger, CSVOutputFormat, TensorBoardOutputFormat
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from imitation_agent.learner import IceHockeyLearner
from imitation_agent.policy import IceHockeyEnv
from imitation_agent.utils import load_policy

# TODO: Add jurgen agent: 'jurgen_agent'
# EXPERT = ['jurgen_agent']
# Expert agent for Offense
EXPERT = ['geoffrey_agent0', 'yann_agent', 'yoshua_agent0']
# Expert agent for Defense
# EXPERT = ['geoffrey_agent1', 'yoshua_agent1']

def main(args):
    rng = np.random.default_rng(0)    

    if not args.only_inference:
        policy_dir = tempfile.TemporaryDirectory(prefix="dagger_policy_")
        print(f"policy_dir: {policy_dir}")
        # create environment
        envs = SubprocVecEnv([lambda: Monitor(IceHockeyLearner(args, logging_level='ERROR')) for _ in range(args.nenv)])
        # experts 
        experts = {key:IceHockeyEnv(envs.observation_space, envs.action_space, key) for key in EXPERT}
        # BC trainer
        bc_trainer = bc.BC(
            observation_space=envs.observation_space,
            action_space=envs.action_space,
            custom_logger=HierarchicalLogger(Logger('./bc_log/', output_formats=[TensorBoardOutputFormat(f'./{args.variant}_bc_log/'), CSVOutputFormat(os.path.join(os.getcwd(),'train_bc_csv.csv'))])),
            rng=rng,
        )

        for epoch in range(1, args.epochs+1):
            print(f"Epoch #{epoch}")
            shuffle(EXPERT)
            for expert_name in EXPERT:
                print (f'epoch: {epoch}, expert: {expert_name}, steps: {int(args.time_steps/len(EXPERT))}')
                # TODO: Use the already trained checkpoint
                # TODO: 'jurgen_agent' requires different environment (observation_space is small)
                # envs = SubprocVecEnv([lambda: Monitor(IceHockeyLearner(args, expert=expert_name, logging_level='ERROR')) for _ in range(args.nenv)])
                expert = experts[expert_name]
                with tempfile.TemporaryDirectory(prefix="dagger_example_") as tmpdir:
                    print(tmpdir)
                    bc_trainer = load_policy(bc_trainer, path=policy_dir.name)
                    dagger_trainer = SimpleDAggerTrainer(
                        venv=envs,
                        scratch_dir=tmpdir,
                        expert_policy=expert,
                        rng=rng,
                        bc_trainer=bc_trainer,
                        custom_logger=HierarchicalLogger(Logger('./dg_log/', output_formats=[TensorBoardOutputFormat(f'./{args.variant}_dg_log/'), CSVOutputFormat(os.path.join(os.getcwd(),'train_dg_csv.csv'))])),
                    )
                    dagger_trainer.train(int(args.time_steps/len(EXPERT)),
                                        rollout_round_min_timesteps=0,
                                        rollout_round_min_episodes=1
                                        )
                    bc_trainer.policy.save(f"{policy_dir.name}/hockey.pt")
        bc_trainer.policy.save(f"./saved_model/{args.variant}.pt")
        policy_dir.cleanup()
        
    print(f"Evaluating")
    args.record_fn=f'{args.variant}.mp4'
    envs_eval = SubprocVecEnv([lambda: Monitor(IceHockeyLearner(args, logging_level='ERROR')) for _ in range(1)])
    expert_eval = IceHockeyEnv(envs_eval.observation_space, envs_eval.action_space, EXPERT[0])
    bc_trainer_eval = bc.BC(
            observation_space=envs_eval.observation_space,
            action_space=envs_eval.action_space,
            # custom_logger=HierarchicalLogger(Logger('./log/',output_formats=[TensorBoardOutputFormat('./dg_log/'), CSVOutputFormat(os.getcwd()+'\\dg_train_csv.csv')])),
            rng=rng,
        )
    # dagger_trainer_eval = SimpleDAggerTrainer(
    #                     venv=envs_eval,
    #                     scratch_dir=None,
    #                     expert_policy=expert_eval,
    #                     rng=rng,
    #                     bc_trainer=bc_trainer_eval,
    #                     # custom_logger=HierarchicalLogger(Logger('./dg_log/', output_formats=[CSVOutputFormat(f'out_infer.csv')])),
    #                 )
    bc_trainer_eval = load_policy(bc_trainer_eval, path='./saved_model/', ckpt=args.variant)
    bc_trainer_eval.policy.eval()
    reward, _ = evaluate_policy(bc_trainer_eval.policy, envs_eval, args.time_steps_infer)
    print("Reward:", reward)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-n', '--nenv', type=int, default=4)
    parser.add_argument('-r', '--record_fn', default=None)
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('-t', '--time_steps', type=int, default=6000)
    parser.add_argument('-e', '--epochs', type=int, default=10)
    parser.add_argument('-ti', '--time_steps_infer', type=int, default=10)
    parser.add_argument('-d', '--deterministic', action='store_true')
    parser.add_argument('--tensor_log', action='store_true')
    parser.add_argument('--debug_mode', action='store_true')
    parser.add_argument('--only_inference', action='store_true')
    parser.add_argument('-v', '--variant', type=str, default='hockey')
    parser.add_argument('--opponent', default='ai')
    parser.add_argument('--use_opponent', action='store_true')

    args = parser.parse_args()
    main(args)
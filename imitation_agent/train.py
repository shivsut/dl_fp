import os, torch
import tempfile, shutil
from argparse import ArgumentParser
from random import shuffle
import numpy as np
import gymnasium as gym
from imitation_local.policies.base import FeedForward32Policy
from imitation_local.util.logger import HierarchicalLogger
from sb3_local.common.evaluation import evaluate_policy
import imitation_local.data.rollout as rollout

from imitation_local.algorithms import bc
from imitation_local.algorithms.dagger import SimpleDAggerTrainer
from imitation_local.policies.serialize import load_policy
from imitation_local.util.util import make_vec_env
from sb3_local.common.logger import Logger, CSVOutputFormat, TensorBoardOutputFormat
from sb3_local.common.monitor import Monitor
from sb3_local.common.vec_env import SubprocVecEnv

from imitation_agent.learner import IceHockeyLearner
from imitation_agent.policy import IceHockeyEnv
from imitation_agent.utils import load_policy
from imitation_agent.algorithms import policy_for_bc

# TODO: Add jurgen agent: 'jurgen_agent'
# EXPERT = ['jurgen_agent']
# Expert agent for Offense
# EXPERT = ['geoffrey_agent0', 'yann_agent', 'yoshua_agent0']
# EXPERT = ['yann_agent']
# Expert agent for Defense
# EXPERT = ['geoffrey_agent1', 'yoshua_agent1']

def main(args):
    rng = np.random.default_rng(0)    

    if not args.only_inference:
        # where all the data will be dumped (checkpoint, video, tensorboard logs)
        policy_dir = tempfile.TemporaryDirectory(prefix="dagger_policy_")
        # Resume training
        if args.resume_training:
            src = os.path.join(os.getcwd(), args.resume_training)
            dst = os.path.join(policy_dir.name, "hockey.pt")
            shutil.copy(src, dst)
            print(f"Resuming the training using ckpt: {src}")
            
        data_dir = os.path.join(os.getcwd(), args.variant)
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        print(f"Data will saved at: {data_dir}")
        # create environment
        # experts = [args.expert]
        envs = SubprocVecEnv([lambda: Monitor(IceHockeyLearner(args, expert=args.expert,logging_level='ERROR')) for _ in range(args.nenv)])
        # experts 
        experts = {key:IceHockeyEnv(envs.observation_space, envs.action_space, key) for key in [args.expert]}

        # BC trainer
        bc_trainer = bc.BC(
            observation_space=envs.observation_space,
            action_space=envs.action_space,
            custom_logger=HierarchicalLogger(Logger(f'{data_dir}/bc_log/', output_formats=[TensorBoardOutputFormat(f'{data_dir}/bc_log/'), CSVOutputFormat(os.path.join(data_dir,'train_bc_csv.csv'))])),
            rng=rng,
            batch_size=args.batch_size,
            # policy=policy_for_bc(observation_space=envs.observation_space, action_space=envs.action_space),
            device=torch.device(args.device),
        )

        for epoch in range(1, args.epochs+1):
            print(f"Epoch #{epoch}")
            shuffle(experts)
            for expert_name in experts:
                print (f'epoch: {epoch}, expert: {expert_name}, steps: {int(args.time_steps/len(experts))}')
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
                        custom_logger=HierarchicalLogger(Logger(f'{data_dir}/dg_log/', output_formats=[CSVOutputFormat(os.path.join(data_dir,'train_dg_csv.csv'))])),
                    )
                    dagger_trainer.train(int(args.time_steps/len(experts)),
                                        rollout_round_min_timesteps=0,
                                        rollout_round_min_episodes=1
                                        )
                    bc_trainer.policy.save(f"{policy_dir.name}/hockey.pt")
        bc_trainer.policy.save(f"{data_dir}/{args.variant}.pt")
        m = torch.jit.script(bc_trainer.policy.mlp_extractor)
        torch.jit.save(m,f"{data_dir}/{args.variant}_jit.pt")
        policy_dir.cleanup()
        
    print(f"Evaluating")
    args.record_fn=f'{data_dir}/{args.variant}.mp4'
    envs_eval = SubprocVecEnv([lambda: Monitor(IceHockeyLearner(args, expert=args.expert,logging_level='ERROR')) for _ in range(1)])
    # expert_eval = IceHockeyEnv(envs_eval.observation_space, envs_eval.action_space, args.expert)
    bc_trainer_eval = bc.BC(
            observation_space=envs_eval.observation_space,
            action_space=envs_eval.action_space,
            rng=rng,
            batch_size=args.batch_size,
            device=torch.device(args.device),
        )
    bc_trainer_eval = load_policy(bc_trainer_eval, path=data_dir, ckpt=args.variant)
    bc_trainer_eval.policy.eval()
    reward, _ = evaluate_policy(bc_trainer_eval.policy, envs_eval, args.time_steps_infer, deterministic=False)
    print("Reward:", reward)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-n', '--nenv', type=int, default=4)
    parser.add_argument('-r', '--record_fn', default=None)
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('-t', '--time_steps', type=int, default=6000)
    parser.add_argument('-e', '--epochs', type=int, default=10)
    parser.add_argument('-ti', '--time_steps_infer', type=int, default=10)
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--tensor_log', action='store_true')
    parser.add_argument('--debug_mode', action='store_true')
    parser.add_argument('--only_inference', action='store_true')
    parser.add_argument('-v', '--variant', type=str, default='hockey')
    parser.add_argument('--opponent', default='ai')
    parser.add_argument('--use_opponent', action='store_true')
    parser.add_argument('--expert', default='yann_agent')
    parser.add_argument('-d', '--discretization', action='store_true')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('-bs', '--batch_size', type=int, default=128)
    parser.add_argument('--resume_training', type=str)

    args = parser.parse_args()
    main(args)
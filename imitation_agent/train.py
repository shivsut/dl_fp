import os, torch
import tempfile, shutil
from argparse import ArgumentParser
from random import shuffle
import numpy as np
import gymnasium as gym
from torch import nn

from imitation_agent.model import IceHockeyModel
from imitation_local.algorithms import bc
from imitation_local.algorithms.dagger import SimpleDAggerTrainer
from imitation_local.util.logger import HierarchicalLogger
from stable_baselines3_local.common import policies, torch_layers
from stable_baselines3_local.common.evaluation import evaluate_policy
from stable_baselines3_local.common.logger import TensorBoardOutputFormat, CSVOutputFormat, Logger
from stable_baselines3_local.common.monitor import Monitor
from stable_baselines3_local.common.policies import BasePolicy

from stable_baselines3_local.common.vec_env import SubprocVecEnv




from imitation_agent.learner import IceHockeyLearner
from imitation_agent.policy import IceHockeyEnv
# from imitation_agent.utils import load_policy, load_model


# from sb3_local.common.policies import ActorCriticPolicy
# from imitation_agent.policies import ActorCriticPolicy


# TODO: Add jurgen agent: 'jurgen_agent'
# EXPERT = ['jurgen_agent']
# Expert agent for Offense
# EXPERT = ['geoffrey_agent0', 'yann_agent', 'yoshua_agent0']
# EXPERT = ['yann_agent']
# Expert agent for Defense
# EXPERT = ['geoffrey_agent1', 'yoshua_agent1']
class FeedForward32Policy(policies.ActorCriticPolicy):
    """A feed forward policy network with two hidden layers of 32 units.

    This matches the IRL policies in the original AIRL paper.

    Note: This differs from stable_baselines3 ActorCriticPolicy in two ways: by
    having 32 rather than 64 units, and by having policy and value networks
    share weights except at the final layer, where there are different linear heads.
    """

    def __init__(self, *args, **kwargs):
        """Builds FeedForward32Policy; arguments passed to `ActorCriticPolicy`."""
        super().__init__(*args, **kwargs)
def main(args):
    rng = np.random.default_rng(0)
    data_dir = os.path.join(os.getcwd(), args.variant)
    envs = SubprocVecEnv(
        [lambda: Monitor(IceHockeyLearner(args, expert=args.expert, logging_level='ERROR', num_env=i)) for i in range(args.nenv)])
    # policy_ac = FeedForward32Policy(
    #     observation_space=envs.observation_space,
    #     action_space=envs.action_space,
    #     lr_schedule=lambda _: torch.finfo(torch.float32).max,
    #     net_arch=[512, 512]
    #     # Set lr_schedule to max value to force error if policy.optimizer
    #     # is used by mistake (should use self.optimizer instead).
    #     # features_extractor_class=extractor
    #
    # )
    if args.act_fn == "tanh":
        activation_function = nn.Tanh
    elif args.act_fn == "relu":
        activation_function = nn.ReLU

    action_logits_dim= envs.action_space.nvec.sum().item()
    policy_ac = IceHockeyModel(
        observation_dim=int(envs.observation_space.shape[0]),
        action_logits_dim=action_logits_dim,
        action_space_dim=int(envs.action_space.shape[0]),
        action_logits_dims_list=envs.action_space.nvec.tolist(),
        lr_scheduler=torch.finfo(torch.float32).max,
        net_arch=[int(x) for x in args.net_arch.split(',')],
        activation_function=activation_function,
        accel_div=args.md,
        use_batch_norm = True if args.batchNorm else False,
    )
    if args.resume_training or args.only_inference:
        path = args.resume_training if args.resume_training else args.only_inference
        src = os.path.join(os.getcwd(), path)
        # dst = os.path.join(policy_dir.name, "hockey.pt")
        # shutil.copy(src, dst)
        print(f"Resuming the training using ckpt: {src}")
        checkpoint = bc.reconstruct_policy(src)
        policy_ac.load_state_dict(checkpoint['state_dict'])
        # policy_ac.load_state_dict(checkpoint['data'])
        policy_ac.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if not args.only_inference:
        # where all the data will be dumped (checkpoint, video, tensorboard logs)
        policy_dir = tempfile.TemporaryDirectory(prefix="dagger_policy_")
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        print(f"Data will saved at: {data_dir}")
        # create environment
        experts = {key:IceHockeyEnv(envs.observation_space, envs.action_space, key, args=args) for key in [args.expert]}
        # BC trainer
        bc_trainer = bc.BC(
            observation_space=envs.observation_space,
            action_space=envs.action_space,
            custom_logger=HierarchicalLogger(Logger(f'{data_dir}/bc_log/', output_formats=[TensorBoardOutputFormat(f'{data_dir}/bc_log/')])),
            rng=rng,
            policy=policy_ac,
            batch_size=args.batch_size,
            device=torch.device(args.device),
            learning_rate=args.lr,
            loss_function=args.lossFn,
        )
        if args.resume_training:
            bc_trainer.optimizer = policy_ac.optimizer
        else:
            policy_ac.optimizer = bc_trainer.optimizer

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
                    # bc_trainer = load_policy(bc_trainer, path=policy_dir.name)
                    dagger_trainer = SimpleDAggerTrainer(
                        venv=envs,
                        model_pt_location=f"{data_dir}/{args.variant}",
                        scratch_dir=tmpdir,
                        expert_policy=expert,
                        rng=rng,
                        bc_trainer=bc_trainer,
                        custom_logger=HierarchicalLogger(Logger(f'{data_dir}/dg_log/', output_formats=[CSVOutputFormat(os.path.join(data_dir,'train_progress.csv'))])),

                    )
                    dagger_trainer.train(int(args.time_steps/len(experts)),
                                        rollout_round_min_timesteps=0,
                                        rollout_round_min_episodes=1,
                                         bc_train_kwargs={'progress_bar':False}
                                        )
                    # bc_trainer._policy.save(f"{policy_dir.name}/hockey.pt")
        bc_trainer._policy.eval()
        m = torch.jit.script(bc_trainer._policy)
        torch.jit.save(m,f"{data_dir}/{args.variant}_jit.pt")
        bc_trainer._policy.save(f"{data_dir}/{args.variant}.pt")
        policy_dir.cleanup()
        
    print(f"Evaluating")
    policy_ac.eval()
    args.record_fn=f'{data_dir}/{args.variant}.mp4'
    envs_eval = SubprocVecEnv([lambda: Monitor(IceHockeyLearner(args, expert=args.expert,logging_level='ERROR', print_episode_result=True)) for _ in range(1)])
    # expert_eval = IceHockeyEnv(envs_eval.observation_space, envs_eval.action_space, args.expert)
    bc_trainer_eval = bc.BC(
            observation_space=envs_eval.observation_space,
            action_space=envs_eval.action_space,
            rng=rng,
            policy=policy_ac,
            batch_size=args.batch_size,
            device=torch.device(args.device),
        )
    # bc_trainer_eval = load_model(IceHockeyModel, path=data_dir, ckpt=args.variant)
    # bc_trainer_eval.policy.eval()
    reward, _ = evaluate_policy(bc_trainer_eval._policy, envs_eval, args.time_steps_infer, deterministic=False)
    print("Reward:", reward)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-n', '--nenv', type=int, default=4)
    parser.add_argument('-r', '--record_fn', default=None)
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('-t', '--time_steps', type=int, default=6000)
    parser.add_argument('-e', '--epochs', type=int, default=1)
    parser.add_argument('-ti', '--time_steps_infer', type=int, default=10)
    parser.add_argument('--only_inference', type=str,)
    parser.add_argument('-v', '--variant', type=str, default='hockey')
    parser.add_argument('--opponent', default='ai')
    parser.add_argument('--use_opponent', action='store_true')
    parser.add_argument('--expert', default='yann_agent')
    parser.add_argument('--md', type=int, required=False)
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('-bs', '--batch_size', type=int, default=128)
    parser.add_argument('--resume_training', type=str)
    parser.add_argument('--net_arch', type=str, default="512,512")
    parser.add_argument('--act_fn', type=str, default="tanh", choices=['tanh', 'relu'])
    parser.add_argument('--team', type=str, default="blue", choices=['blue', 'red'])
    parser.add_argument('--batchNorm', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lossFn', type=str, default="entropy", choices=['entropy', 'mse', 'huber'])

    args = parser.parse_args()
    main(args)
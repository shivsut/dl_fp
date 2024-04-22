import os, torch
import tempfile
from argparse import ArgumentParser
import numpy as np
from torch import nn

from state_agent.model import IceHockeyModel
from imitation_local.algorithms import bc
from imitation_local.util.logger import HierarchicalLogger
from stable_baselines3_local.common import policies
from stable_baselines3_local.common.logger import TensorBoardOutputFormat, Logger
from stable_baselines3_local.common.monitor import Monitor

from stable_baselines3_local.common.vec_env import SubprocVecEnv
from imitation_agent.learner import IceHockeyLearner


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
        print(f"Resuming the training using ckpt: {src}")
        checkpoint = bc.reconstruct_policy(src)
        policy_ac = policy_ac.to(args.device)
        policy_ac.load_state_dict(checkpoint['state_dict'])
        policy_ac.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if not args.only_inference:
        # where all the data will be dumped (checkpoint, video, tensorboard logs)
        policy_dir = tempfile.TemporaryDirectory(prefix="dagger_policy_")
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        print(f"Data will saved at: {data_dir}")

        bc_trainer = bc.BC(
            observation_space=envs.observation_space,
            action_space=envs.action_space,
            custom_logger=HierarchicalLogger(Logger(f'{data_dir}/bc_log/', output_formats=[TensorBoardOutputFormat(f'{data_dir}/bc_log/')])),
            rng=rng,
            policy=policy_ac,
            batch_size=args.batch_size,
            device=torch.device(args.device),
            learning_rate=args.lr,
        )
        if args.resume_training:
            bc_trainer.optimizer = policy_ac.optimizer
        else:
            policy_ac.optimizer = bc_trainer.optimizer

        bc_trainer._policy.eval()
        m = torch.jit.script(bc_trainer._policy)
        torch.jit.save(m,f"{data_dir}/{args.variant}_jit.pt")
        policy_dir.cleanup()
        


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

    args = parser.parse_args()
    main(args)
import tempfile
from argparse import ArgumentParser

import numpy as np
import gymnasium as gym
from stable_baselines3.common.evaluation import evaluate_policy
import imitation.data.rollout as rollout

from imitation.algorithms import bc
from imitation.algorithms.dagger import SimpleDAggerTrainer
from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env

from reward_env import IceHockeyEnvImitation

from policy_env import IceHockeyEnv

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
env = IceHockeyEnvImitation(args, logging_level='ERROR')
expert = IceHockeyEnv(env.observation_space, env.action_space)
# expert = IceHockeyEnv()
# expert = load_policy(
#     "ppo-huggingface",
#     organization="HumanCompatibleAI",
#     env_name="seals-CartPole-v0",
#     venv=env,
# )

bc_trainer = bc.BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    rng=rng,
)
with tempfile.TemporaryDirectory(prefix="dagger_example_") as tmpdir:
    print(tmpdir)
    dagger_trainer = SimpleDAggerTrainer(
        venv=env,
        scratch_dir=tmpdir,
        expert_policy=expert,
        bc_trainer=bc_trainer,
        rng=rng,
    )
    dagger_trainer.train(1000)

dagger_trainer.save_trainer()
reward, _ = evaluate_policy(dagger_trainer.policy, env, 10)
print("Reward:", reward)
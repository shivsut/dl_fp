from typing import Callable, Tuple, Union, List

import numpy as np
import torch
from torch import nn

from stable_baselines3_local.common.distributions import Distribution, MultiCategoricalDistribution

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class IceHockeyModel(nn.Module):
    def __init__(self,
                 observation_dim: int,
                 action_logits_dim: int,
                 action_logits_dims_list: [int],
                 action_space_dim: int,
                 lr_scheduler: float,
                 net_arch: [int],
                 activation_function: nn.Module = nn.Tanh,
                 ortho_init: bool = True,  # TODO optimize
                 log_std_init: float = 0.0,
                 full_std: bool = True,
                 use_expln: bool = False,
                 squash_output: bool = False,
                 accel_div: int = 100,
                 use_batch_norm=False):

        super(IceHockeyModel, self).__init__()
        self.observation_dim = observation_dim
        self.action_logits_dim = action_logits_dim
        self.action_space_dim = action_space_dim
        self.action_logits_dims_list =action_logits_dims_list
        self.net_arch = net_arch
        self.activation_function = activation_function
        self.ortho_init = ortho_init
        self.log_std_init = log_std_init
        self.full_std = full_std
        self.use_expln = use_expln
        self.squash_output = squash_output
        self.lr_scheduler = lr_scheduler
        self.accel_div = float(accel_div)

        self.policy_nn = nn.Sequential()
        self.value_nn = nn.Sequential()
        self.action_nn = nn.Sequential()

        prev_layer_dim = observation_dim
        for layer in self.net_arch:
            self.policy_nn.append(nn.Linear(prev_layer_dim, layer))
            if use_batch_norm:
                self.policy_nn.append(nn.BatchNorm1d(layer))
            # self.value_nn.append(nn.Linear(prev_layer_dim, layer))
            self.policy_nn.append(self.activation_function())
            # self.value_nn.append(self.activation_function())
            prev_layer_dim = layer

        # self.value_net2 = nn.Linear(prev_layer_dim, 1)
        self.policy_nn.append(nn.Linear(prev_layer_dim, self.action_logits_dim))

        self.device = device

        # TODO ortho_init
        if self.training:
            self.init_dist()
        # self.distribution_infer = Distribution(self.action_logits_dims_list)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr_scheduler)


    def predict(self, observation: np.ndarray, state : np.ndarray=None, episode_start: np.ndarray=None, deterministic:bool=False):
        observation = torch.tensor(observation)
        observation = observation.to(torch.float32)
        self.train(False)
        with torch.no_grad():
            actions = self.predict_action(observation, deterministic)
        actions.cpu().numpy().reshape((-1, self.action_space_dim))
        return actions, state
    # @torch.jit.script
    def forward(self, observation):
        # observation = torch.tensor(observation)
        # observation = observation.to(self.device)
        policy_output = self.policy_nn(observation)
        # action_output = self.action_nn(policy_output)
        # action_output = action_output.to("cpu")
        res = []
        for split in torch.split(policy_output, self.action_logits_dims_list):
            new_split = split - split.logsumexp(dim=-1, keepdim=True)
            probs = torch.nn.functional.softmax(new_split, dim=-1)
            out = torch.argmax(probs, dim=-1)
            out_new = out.to(torch.float32)
            res.append(out_new)
        res[0] /= self.accel_div
        res[1] -= 1.0
        return res
    def predict_action(self, observation: torch.Tensor, deterministic=False) -> torch.Tensor:
        observation = observation.to(self.device)
        policy_output = self.policy_nn(observation)
        # action_output = self.action_nn(policy_output)
        action_output = policy_output.to("cpu")
        actions = self.distribution.proba_distribution(action_output)
        return actions.mode() if deterministic else actions.sample()

    @torch.jit.ignore
    def init_dist(self):
        self.distribution = MultiCategoricalDistribution(self.action_logits_dims_list)

    @torch.jit.ignore
    def evaluate_actions(self, observation: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        observation = nn.Flatten()(observation).to(torch.float32)
        policy_output = self.policy_nn(observation)
        # value_output = self.value_nn(observation)
        # actions_output = self.action_nn(policy_output)
        log_probability = self.distribution.proba_distribution(policy_output).log_prob(actions)
        return None, log_probability, self.distribution.entropy()


        # return actions.mode() if deterministic else actions.sample()

    def _get_constructor_parameters(self):
        data = {}

        data.update(
            dict(
                observation_dim=self.observation_dim,
                action_logits_dim=self.action_logits_dim,
                action_space_dim=self.action_space_dim,
                action_logits_dims_list=self.action_logits_dims_list,
                net_arch=self.net_arch,
                activation_function=self.activation_function,
                log_std_init=self.log_std_init,
                lr_schedule=0.0,  # dummy lr schedule, not needed for loading policy alone
                ortho_init=self.ortho_init,
                accel_div=self.accel_div
            )
        )
        return data

    def save(self, path: str) -> None:
        torch.save({"state_dict": self.state_dict(), "data": self._get_constructor_parameters()}, path)


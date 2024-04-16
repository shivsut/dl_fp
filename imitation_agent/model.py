from typing import Callable, Tuple

import numpy as np
import torch
from torch import nn

from imitation_agent.categorical import Categorical

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# @torch.jit.script
class Distribution(nn.Module):
    def __init__(self, dimension : int):
        super(Distribution, self).__init__()
        self.dimension = dimension
        self.dummy = torch.Tensor((1,1))


    def create_prob_distribution(self, logits: torch.Tensor):
        self.distribution = [Categorical(probs=None, logits=split, validate_args=None) for split in torch.split(logits, self.dimension, dim=1)]
        return self

    def log_probability(self, actions: torch.Tensor):
        return torch.stack([dist.log_prob(action) for dist, action in zip(self.distribution, torch.unbind(actions, dim=1))], dim=1).sum(dim=1)

    def entropy(self) -> torch.Tensor:
        return torch.stack([dist.entropy() for dist in self.distribution], dim=1).sum(dim=1)

    def sample(self) -> torch.Tensor:
        return torch.stack([dist.sample() for dist in self.distribution], dim=1)

    def mode(self) -> torch.Tensor:
        return torch.stack([torch.argmax(dist.probs, dim=1) for dist in self.distribution], dim=1)

    # def actions_from_params(self, action_logits: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
    #     # Update the proba distribution
    #     self.create_prob_distribution(action_logits)
    #     return self.get_actions(deterministic=deterministic)
    #
    # def log_prob_from_params(self, action_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    #     actions = self.actions_from_params(action_logits)
    #     log_prob = self.log_probability(actions)
    #     return actions, log_prob

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
                 optimizer_class: torch.optim.Optimizer = torch.optim.Adam):

        super(IceHockeyModel, self).__init__()
        self.observation_dim = observation_dim
        self.action_logits_dim = action_logits_dim
        self.action_space_dim = action_space_dim
        self.action_logits_dims_list =action_logits_dims_list
        self.optimizer_class = optimizer_class
        self.net_arch = net_arch
        self.activation_function = activation_function
        self.ortho_init = ortho_init
        self.log_std_init = log_std_init
        self.full_std = full_std
        self.use_expln = use_expln
        self.squash_output = squash_output
        self.lr_scheduler = lr_scheduler

        self.distribution = Distribution(self.action_logits_dims_list)

        self.policy_nn = nn.Sequential()
        self.value_nn = nn.Sequential()
        self.action_nn = nn.Sequential()

        prev_layer_dim = observation_dim
        for layer in self.net_arch:
            self.policy_nn.append(nn.Linear(prev_layer_dim, layer))
            self.value_nn.append(nn.Linear(prev_layer_dim, layer))
            self.policy_nn.append(self.activation_function())
            self.value_nn.append(self.activation_function())
            prev_layer_dim = layer

        self.value_net2 = nn.Linear(prev_layer_dim, 1)
        self.action_nn.append(nn.Linear(prev_layer_dim, self.action_logits_dim))

        self.device = device

        # TODO ortho_init

        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_scheduler)

    def forward(self, observation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        policy_output = self.policy_nn(observation)
        value_output = self.value_nn(observation)
        # actions_output = self.action_nn(observation)
        actions_output = self.distribution.create_prob_distribution(policy_output).mode()
        log_probability = self.distribution.log_probability(actions_output)
        values = self.value_net2(value_output)
        return actions_output.reshape((-1, *self.action_logits_dim)), values, log_probability

    def evaluate_actions(self, observation: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        observation = nn.Flatten()(observation).to(torch.float32)
        policy_output = self.policy_nn(observation)
        value_output = self.value_nn(observation)
        actions_output = self.action_nn(policy_output)
        log_probability = self.distribution.create_prob_distribution(actions_output).log_probability(actions)
        return self.value_net2(value_output), log_probability, self.distribution.entropy()

    def predict(self, observation: np.ndarray, state : np.ndarray=None, episode_start: np.ndarray=None, deterministic:bool=True):
        observation = torch.tensor(observation)
        observation = observation.to(torch.float32)
        self.train(False)
        with torch.no_grad():
            actions = self.predict_action(observation)
        actions.cpu().numpy().reshape((-1, self.action_space_dim))
        return actions, state

    def predict_action(self, observation: torch.Tensor) -> torch.Tensor:
        policy_output = self.policy_nn(observation)
        action_output = self.action_nn(policy_output)
        actions = self.distribution.create_prob_distribution(action_output).mode()
        return actions

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
                optimizer_class=self.optimizer_class
            )
        )
        return data

    def save(self, path: str) -> None:
        torch.save({"state_dict": self.state_dict(), "data": self._get_constructor_parameters()}, path)


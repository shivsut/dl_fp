from typing import Callable, Tuple

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Distribution:
    def __init__(self, dimension : [int]):
        self.dimension = dimension

    def create_prob_distribution(self, logits: torch.Tensor):
        self.distribution = [Categorical(logits=split) for split in torch.split(logits, list(self.dimension), dim=1)]
        return self

    def log_probability(self, actions: torch.Tensor):
        return torch.stack([dist.log_prob(action) for dist, action in zip(self.distribution, torch.unbind(actions, dim=1))], dim=1).sum(dim=1)

    def entropy(self) -> torch.Tensor:
        return torch.stack([dist.entropy() for dist in self.distribution], dim=1).sum(dim=1)

    def sample(self) -> torch.Tensor:
        return torch.stack([dist.sample() for dist in self.distribution], dim=1)

    def mode(self) -> torch.Tensor:
        return torch.stack([torch.argmax(dist.probs, dim=1) for dist in self.distribution], dim=1)

    def actions_from_params(self, action_logits: torch.Tensor, deterministic: bool = False) -> th.Tensor:
        # Update the proba distribution
        self.create_prob_distribution(action_logits)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, action_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        actions = self.actions_from_params(action_logits)
        log_prob = self.log_probability(actions)
        return actions, log_prob

class IceHockeyModel(nn.Module):
    def __init__(self,
                observation_space: np.ndarray,
                action_space: np.ndarray,
                lr_scheduler: Callable[[float], float],
                net_arch: [int],
                activation_function: nn.Module = nn.Tanh,
                ortho_init: bool = True, # TODO optimize
                log_std_init: float = 0.0,
                full_std: bool = True,
                use_expln: bool = False,
                squash_output: bool = False,
                optimizer_class: torch.optim.Optimizer = torch.optim.Adam):

        super(IceHockeyModel, self).__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.optimizer_class = optimizer_class
        self.net_arch = net_arch
        self.activation_function = activation_function
        self.ortho_init = ortho_init
        self.log_std_init = log_std_init
        self.full_std = full_std
        self.use_expln = use_expln
        self.squash_output = squash_output

        self.distribution = Distribution(list(self.action_space))

        self.policy_nn = nn.Sequential()
        self.value_nn = nn.Sequential()
        self.action_nn = nn.Sequential()

        prev_layer_dim = observation_space.shape[0]
        for layer in self.net_arch:
            self.policy_nn.append(nn.Linear(prev_layer_dim, layer))
            self.value_nn.append(nn.Linear(prev_layer_dim, layer))
            self.policy_nn.append(self.activation_function())
            self.value_nn.append(self.activation_function())
            prev_layer_dim = layer

        self.value_net2 = nn.Linear(prev_layer_dim, 1)
        self.action_nn.append(nn.Linear(prev_layer_dim, self.action_space.shape[0]))

        # TODO ortho_init

        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_scheduler(1))

    def forward(self, observation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        policy_output = self.policy_nn(observation)
        value_output = self.value_nn(observation)
        # actions_output = self.action_nn(observation)
        actions_output = self.distribution.create_prob_distribution(policy_output).mode()
        log_probability = self.distribution.log_probability(actions_output)
        values = self.value_net2(value_output)
        return actions_output.reshape((-1, *self.action_space.shape)), values, log_probability

    def evaluate_actions(self, observation: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        policy_output = self.policy_nn(observation)
        value_output = self.value_nn(observation)
        actions_output = self.action_nn(observation)
        log_probability = self.distribution.create_prob_distribution(policy_output).log_probability(actions_output)
        values = self.value_net2(value_output)
        entropy = self.distribution.entropy()
        return values, log_probability, entropy

    def predict(self, observation: torch.Tensor) -> torch.Tensor:
        policy_output = self.policy_nn(observation)
        action_output = self.action_nn(policy_output)
        actions = self.distribution.create_prob_distribution(action_output).mode()
        return  actions



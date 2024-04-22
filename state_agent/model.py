
import torch
from torch import nn

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
                 accel_div: int = 100,
                 use_batch_norm=False, 
                 learning_rate = 1.0):

        super(IceHockeyModel, self).__init__()
        self.observation_dim = observation_dim
        self.action_logits_dim = action_logits_dim
        self.action_space_dim = action_space_dim
        self.action_logits_dims_list =action_logits_dims_list
        self.net_arch = net_arch
        self.activation_function = activation_function
        self.ortho_init = ortho_init
        self.lr_scheduler = lr_scheduler
        self.accel_div = float(accel_div)

        self.policy_nn = nn.Sequential()
        self.use_batch_norm = use_batch_norm

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
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)


    def forward(self, observation):
        policy_output = self.policy_nn(observation)
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


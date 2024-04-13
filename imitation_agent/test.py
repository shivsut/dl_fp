import torch
from torch.distributions import Categorical

intervals = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]
probabilities = [0.2, 0.5, 0.3]
interval_probs = torch.tensor([prob[1] - prob[0] for prob in intervals]) * torch.tensor(probabilities)
print (interval_probs)
dist = Categorical(interval_probs)
sampled_index = dist.sample()
print (sampled_index)
print(torch.tensor((intervals[sampled_index][0] + intervals[sampled_index][1]) / 2.0))
from rlvf.model import HyperNetwork
from rlvf.ppo import PPO
import torch

hn = HyperNetwork(5, ["q", "v"], dims={"q": [4096, 4096], "v": [4096, 1024]}, profile_dim=19, rank=8)

input = torch.arange(19, dtype=torch.float).expand(2, -1)

print(hn.get_action_and_logprob(input))

pytorch_total_params = sum(p.numel() for p in hn.parameters() if p.requires_grad)
print(pytorch_total_params)

print(hn.get_value(input).shape)

action, logprob, A, B = hn.get_action_and_logprob(input)

print(action.shape)

lp, entropy = hn.get_action_and_logprob(input, action=action, use_action=True)

print(entropy)

print(logprob.shape, lp.shape, entropy.shape)
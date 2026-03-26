from rlvf.model import HyperNetwork
import torch

hn = HyperNetwork(5, ["q", "v"], dims={"q": [4096, 4096], "v": [4096, 1024]}, profile_dim=19, rank=8)

input = torch.arange(19).unsqueeze(0)

print(hn.get_action_and_logprob(input))

pytorch_total_params = sum(p.numel() for p in hn.parameters() if p.requires_grad)
print(pytorch_total_params)
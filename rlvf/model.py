import torch
from torch.distributions import Normal, Independent
"""
ALL ARCHITECTURE IN HERE IS MOSTLY INSPIRED BY, ADAPTED AND SIMPLIFIED FROM ZHyper

https://arxiv.org/pdf/2510.19733

"""

class Mixer(torch.nn.Module):
    def __init__(self, mlp_in):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Dropout(0.05),
            torch.nn.Linear(mlp_in, mlp_in * 4),
            torch.nn.SiLU(),
            torch.nn.Dropout(0.05),
            torch.nn.Linear(mlp_in * 4, mlp_in),
            torch.nn.SiLU(),
            torch.nn.Dropout(0.05)
        )
    def forward(self, x):
        return self.layers(x)
    

class MLPResidual(torch.nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim),
            torch.nn.Linear(in_dim, in_dim * 4),
            torch.nn.SiLU(),
            torch.nn.Dropout(0.05),
            torch.nn.Linear(in_dim * 4, in_dim),
            torch.nn.SiLU(),
            torch.nn.Dropout(0.05),
        )
    def forward(self, x):
        return x + self.layers(x)

class MLPProjection(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim),
            torch.nn.Linear(in_dim, in_dim * 4),
            torch.nn.SiLU(),
            torch.nn.Dropout(0.05),
            torch.nn.Linear(in_dim * 4, out_dim),
            torch.nn.SiLU(),
        )
    def forward(self, x):
        return self.layers(x)

class HyperNetwork(torch.nn.Module):
    def __init__(self, num_layers=5, layer_types=["q", "v"], dims={"q": [10, 20], "v": [10, 20]}, emb_l_dim=32, emb_t_dim=32, profile_dim=19, rank=8, head_in_dim=512):
        super().__init__()

        self.num_layers = num_layers
        self.num_types = len(layer_types)
        self.layer_types = layer_types
        self.rank = rank
        self.action_dim = num_layers * len(layer_types) * rank  # 32 * 2 * 8 = 512

        # Precomputed layer and type batches
        layer_indices = torch.arange(self.num_layers)
        type_indices = torch.arange(self.num_types)
        l, t = torch.meshgrid(layer_indices, type_indices, indexing="ij")
        self.register_buffer("l", l.flatten())
        self.register_buffer("t", t.flatten())

        # EMBEDDINGS
        self.layer_emb = torch.nn.Embedding(self.num_layers, emb_l_dim)
        self.type_emb = torch.nn.Embedding(self.num_types, emb_t_dim)
        context_dim = profile_dim + emb_l_dim + emb_t_dim
        self.mixer = Mixer(mlp_in=context_dim)

        # MLPS
        self.mlp1 = MLPResidual(context_dim)
        self.mlp2 = MLPResidual(context_dim)
        self.mlp3 = MLPProjection(in_dim=context_dim, out_dim=head_in_dim)

        # A and B
        self.A = torch.nn.ParameterDict(
            {k: torch.nn.Parameter(torch.empty(num_layers, rank, dims[k][1])) for k in layer_types}
        )
        for _, A in self.A.items():
            torch.nn.init.kaiming_uniform_(A, a=0, mode="fan_in")

        self.B = torch.nn.ParameterDict(
            {k: torch.nn.Parameter(torch.nn.init.normal_(
                torch.empty(num_layers, dims[k][0], rank), mean=0.0, std=0.05
            )) for k in layer_types}
        )

        # HEAD
        self.head_mean = torch.nn.Linear(head_in_dim, rank)
        self.head_log_std = torch.nn.Linear(head_in_dim, rank)

        torch.nn.init.orthogonal_(self.head_mean.weight, gain=0.01)
        torch.nn.init.zeros_(self.head_mean.bias)
        torch.nn.init.constant_(self.head_log_std.weight, 0)
        torch.nn.init.constant_(self.head_log_std.bias, -1.0)

        # VALUE HEAD
        self.value_head = torch.nn.Sequential(
            torch.nn.Linear(profile_dim, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 1)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.unsqueeze(1).expand(-1, self.num_layers * self.num_types, -1)
        l = self.l.unsqueeze(0).expand(batch_size, -1)
        t = self.t.unsqueeze(0).expand(batch_size, -1)

        emb_l = self.layer_emb(l)
        emb_t = self.type_emb(t)
        x = torch.concat([x, emb_l, emb_t], dim=-1)
        x = self.mixer(x)

        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.mlp3(x)

        z_mean = self.head_mean(x)
        z_logstd = self.head_log_std(x)
        z_logstd = torch.clamp(z_logstd, -20, 2)

        return z_mean, z_logstd

    def get_action_and_logprob(self, x, action=None, use_action=False):
        z_mean, z_logstd = self.forward(x)
        z_std = z_logstd.exp()
        dist = Independent(Normal(z_mean, z_std), 2)

        if use_action:
            # action shape: (batch_size, num_layers*num_types, rank)
            log_prob = dist.log_prob(action) / self.action_dim  # (batch_size,)
            return log_prob, dist.entropy().mean()

        z = dist.rsample()
        log_prob = dist.log_prob(z) / self.action_dim  # (batch_size,)
        return z, log_prob, self.A, self.B

    def get_value(self, x):
        return self.value_head(x)
import torch
from torch.distributions import Normal, Independent
"""
Architecture inspired by, adapted and simplified from ZHyper
https://arxiv.org/pdf/2510.19733
 
Adapter generation follows VeRA (Kopiczko et al., ICLR 2024)
https://arxiv.org/abs/2310.11454
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

        # ── Frozen random A and B (VeRA-style) ────────────────────────────────
        # Kaiming uniform init as in the VeRA paper.
        # requires_grad=False, these are never updated by the optimizer.
        self.A = torch.nn.ParameterDict(
            {
                k: torch.nn.Parameter(
                    torch.empty(num_layers, rank, dims[k][1]), requires_grad=False
                )
                for k in layer_types
            }
        )
        for _, A in self.A.items():
            torch.nn.init.kaiming_uniform_(A)
 
        self.B = torch.nn.ParameterDict(
            {
                k: torch.nn.Parameter(
                    torch.empty(num_layers, dims[k][0], rank), requires_grad=False
                )
                for k in layer_types
            }
        )
        for _, B in self.B.items():
            torch.nn.init.kaiming_uniform_(B)


        # HEAD

        self.b_head_mean = torch.nn.Linear(head_in_dim, rank)
        self.d_head_mean = torch.nn.Linear(head_in_dim, rank)
        torch.nn.init.orthogonal_(self.b_head_mean.weight, gain=0.01)
        torch.nn.init.zeros_(self.b_head_mean.bias)

        torch.nn.init.orthogonal_(self.d_head_mean.weight, gain=0.01)
        torch.nn.init.zeros_(self.d_head_mean.bias)

        self.log_std_d = torch.nn.Parameter(
            torch.full((self.num_layers * self.num_types, rank), -1.0)
        )
        self.log_std_b = torch.nn.Parameter(
           torch.full((self.num_layers * self.num_types, rank), -1.0)
        )

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

        b_mean = self.b_head_mean(x)
        b_logstd = torch.clamp(self.log_std_b, -3, -1).unsqueeze(0).expand(batch_size, -1, -1)

        d_mean = self.d_head_mean(x)
        d_logstd = torch.clamp(self.log_std_d, -3, -1).unsqueeze(0).expand(batch_size, -1, -1)

        return (b_mean, d_mean), (b_logstd, d_logstd)

    def get_action_and_logprob(self, x, action=None, use_action=False):
        means, logstds = self.forward(x)
        b_mean, d_mean = means
        b_log_std, d_log_std = logstds

        b_std = b_log_std.exp()
        dist_b = Independent(Normal(b_mean, b_std), 2)

        d_std = d_log_std.exp()
        dist_d = Independent(Normal(d_mean, d_std), 2)

        if use_action:
            # action shape: (2, batch_size, num_layers*num_types, rank)
            b, d = action
            log_prob_b = dist_b.log_prob(b)  # (batch_size,)
            log_prob_d = dist_d.log_prob(d)
            log_prob = (log_prob_b + log_prob_d) / (2*self.action_dim)

            entropy = (dist_b.entropy() + dist_d.entropy()).mean()
            return log_prob, entropy

        b = dist_b.rsample()
        d = dist_d.rsample()

        log_prob_b = dist_b.log_prob(b)  # (batch_size,)
        log_prob_d = dist_d.log_prob(d)
        log_prob = (log_prob_b + log_prob_d) / (2*self.action_dim)

        return (b,d), log_prob, self.A, self.B

    def get_value(self, x):
        return self.value_head(x)
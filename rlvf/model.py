import torch
from torch.distributions import Normal, Independent
"""
Architecture inspired by, adapted and simplified from ZHyper
https://arxiv.org/pdf/2510.19733

Adapter generation follows VeRA (Kopiczko et al., ICLR 2024)
https://arxiv.org/abs/2310.11454

PARAMETERIZATION (paper-faithful VeRA)
--------------------------------------
    ΔW = Λ_b · B · Λ_d · A
with, per adapted layer & type:
    d ∈ R^rank      (scales rows of A)
    b ∈ R^{d_out}   (scales rows of the final matrix — OUTPUT dimension)
Because b acts on output coordinates, it is NOT absorbable into d: no
degeneracy (unlike the earlier b, d ∈ R^rank variant where only b∘d mattered).

Init follows the paper: b-mean starts at ~0 ("the weight matrix is unaffected
during the first forward pass"), d-mean starts at D_INIT = 0.1 (paper default;
the paper's ablation finds 1e-7 also works, 1.0 does not).

ACTION SPACE WARNING
--------------------
The stochastic action is (b, d):
    b["q"]: (num_layers, 4096), b["v"]: (num_layers, 1024),
    d: (num_layers*types, rank)
Total action dim = num_layers*(d_out_q + d_out_v) + num_layers*types*rank
                 = 32*(4096+1024) + 512 = 164,352 for the default config.
Joint log-probs / KL / entropy all scale with this. See train.py notes.

Everything else from the previous fix round is retained: joint log-prob (no
normalization), soft-bounded log_std (raw 0 -> -2.0, smooth gradient),
deterministic eval mode, dropout=0 in the policy trunk.
"""

LOG_STD_MIN = -4.0
LOG_STD_MAX = 0.0   # midpoint = -2.0
D_INIT = 0.1        # VeRA paper default init for the d vector


class Mixer(torch.nn.Module):
    def __init__(self, mlp_in, dropout=0.0):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.Linear(mlp_in, mlp_in * 4),
            torch.nn.SiLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(mlp_in * 4, mlp_in),
            torch.nn.SiLU(),
            torch.nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.layers(x)


class MLPResidual(torch.nn.Module):
    def __init__(self, in_dim, dropout=0.0):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim),
            torch.nn.Linear(in_dim, in_dim * 4),
            torch.nn.SiLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(in_dim * 4, in_dim),
            torch.nn.SiLU(),
            torch.nn.Dropout(dropout),
        )
    def forward(self, x):
        return x + self.layers(x)


class MLPProjection(torch.nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim),
            torch.nn.Linear(in_dim, in_dim * 4),
            torch.nn.SiLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(in_dim * 4, out_dim),
            torch.nn.SiLU(),
        )
    def forward(self, x):
        return self.layers(x)


class HyperNetwork(torch.nn.Module):
    def __init__(self, num_layers=5, layer_types=["q", "v"],
                 dims={"q": [10, 20], "v": [10, 20]},
                 emb_l_dim=32, emb_t_dim=32, profile_dim=19,
                 rank=8, head_in_dim=512, dropout=0.0):
        super().__init__()

        self.num_layers = num_layers
        self.num_types = len(layer_types)
        self.layer_types = layer_types
        self.rank = rank
        self.dims = dims
        # d_out per type (rows of the adapted weight matrix = dims[k][0])
        self.d_out = {k: dims[k][0] for k in layer_types}
        self.action_dim = (
            num_layers * sum(self.d_out[k] for k in layer_types)   # b
            + num_layers * self.num_types * rank                   # d
        )

        # Precomputed layer and type batches
        layer_indices = torch.arange(self.num_layers)
        type_indices = torch.arange(self.num_types)
        l, t = torch.meshgrid(layer_indices, type_indices, indexing="ij")
        self.register_buffer("l", l.flatten())
        self.register_buffer("t", t.flatten())
        # boolean masks over the L*T position axis, one per type
        for ti, k in enumerate(layer_types):
            self.register_buffer(f"pos_mask_{k}", (t.flatten() == ti))

        # EMBEDDINGS
        self.layer_emb = torch.nn.Embedding(self.num_layers, emb_l_dim)
        self.type_emb = torch.nn.Embedding(self.num_types, emb_t_dim)
        context_dim = profile_dim + emb_l_dim + emb_t_dim
        self.mixer = Mixer(mlp_in=context_dim, dropout=dropout)

        # MLPS
        self.mlp1 = MLPResidual(context_dim, dropout=dropout)
        self.mlp2 = MLPResidual(context_dim, dropout=dropout)
        self.mlp3 = MLPProjection(in_dim=context_dim, out_dim=head_in_dim,
                                  dropout=dropout)

        # ── Frozen random A and B (VeRA-style) ────────────────────────────────
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

        # HEADS
        # d-head: shared across (layer, type) positions, outputs rank values.
        # Bias init D_INIT per the VeRA paper (d starts at a nonzero constant).
        self.d_head_mean = torch.nn.Linear(head_in_dim, rank)
        torch.nn.init.orthogonal_(self.d_head_mean.weight, gain=0.001)
        torch.nn.init.constant_(self.d_head_mean.bias, D_INIT)

        # b-heads: one per type (output dims differ, e.g. 4096 q / 1024 v).
        # Zero-ish init per the paper: mean ΔW ≈ 0 at the first forward pass.
        self.b_head_mean = torch.nn.ModuleDict()
        for k in layer_types:
            head = torch.nn.Linear(head_in_dim, self.d_out[k])
            torch.nn.init.orthogonal_(head.weight, gain=0.001)
            torch.nn.init.zeros_(head.bias)
            self.b_head_mean[k] = head

        # Raw log_std params; 0.0 maps to log_std = -2.0 (see _soft_bound).
        self.log_std_raw_d = torch.nn.Parameter(
            torch.zeros(self.num_layers * self.num_types, rank)
        )
        self.log_std_raw_b = torch.nn.ParameterDict(
            {
                k: torch.nn.Parameter(torch.zeros(num_layers, self.d_out[k]))
                for k in layer_types
            }
        )

        # VALUE HEAD
        self.value_head = torch.nn.Sequential(
            torch.nn.Linear(profile_dim, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 1)
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _soft_bound(raw):
        """Smooth map R -> [LOG_STD_MIN, LOG_STD_MAX]; raw=0 -> midpoint."""
        mid = 0.5 * (LOG_STD_MAX + LOG_STD_MIN)
        half = 0.5 * (LOG_STD_MAX - LOG_STD_MIN)
        return mid + half * torch.tanh(raw)

    def _log_std_d(self):
        return self._soft_bound(self.log_std_raw_d)

    def _log_std_b(self, k):
        return self._soft_bound(self.log_std_raw_b[k])

    def log_std_mean(self):
        """Scalar diagnostic: mean log_std over all action dims."""
        parts = [self._log_std_d().flatten()]
        parts += [self._log_std_b(k).flatten() for k in self.layer_types]
        return torch.cat(parts).mean()

    def _trunk(self, x):
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
        return x                                  # (B, L*T, head_in_dim)

    def forward(self, x):
        """
        Returns
        -------
        d_mean : (B, L*T, rank)
        b_mean : dict k -> (B, L, d_out[k])
        """
        h = self._trunk(x)
        d_mean = self.d_head_mean(h)              # (B, L*T, rank)
        b_mean = {}
        for k in self.layer_types:
            mask = getattr(self, f"pos_mask_{k}")     # (L*T,)
            h_k = h[:, mask, :]                       # (B, L, head_in)
            b_mean[k] = self.b_head_mean[k](h_k)      # (B, L, d_out[k])
        return d_mean, b_mean

    def _dists(self, x):
        d_mean, b_mean = self.forward(x)
        B_ = x.shape[0]
        dist_d = Independent(
            Normal(d_mean,
                   self._log_std_d().exp().unsqueeze(0).expand(B_, -1, -1)), 2)
        dist_b = {
            k: Independent(
                Normal(b_mean[k],
                       self._log_std_b(k).exp().unsqueeze(0).expand(B_, -1, -1)), 2)
            for k in self.layer_types
        }
        return dist_d, dist_b

    def get_action_and_logprob(self, x, action=None, use_action=False,
                               deterministic=False):
        """
        Action format: (b, d) with
            b : dict k -> (B, num_layers, d_out[k])
            d : (B, num_layers*num_types, rank)
        Log-prob is the JOINT log density over all of b and d (no
        normalization — the PPO ratio must be the true importance ratio).
        """
        dist_d, dist_b = self._dists(x)

        if use_action:
            b, d = action
            log_prob = dist_d.log_prob(d)
            entropy = dist_d.entropy().mean()
            for k in self.layer_types:
                log_prob = log_prob + dist_b[k].log_prob(b[k])
                entropy = entropy + dist_b[k].entropy().mean()
            return log_prob, entropy

        if deterministic:
            d = dist_d.base_dist.loc
            b = {k: dist_b[k].base_dist.loc for k in self.layer_types}
        else:
            d = dist_d.rsample()
            b = {k: dist_b[k].rsample() for k in self.layer_types}

        log_prob = dist_d.log_prob(d)
        for k in self.layer_types:
            log_prob = log_prob + dist_b[k].log_prob(b[k])

        return (b, d), log_prob, self.A, self.B

    def get_value(self, x):
        return self.value_head(x)
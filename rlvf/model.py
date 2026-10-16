import torch
"""
Architecture inspired by, adapted and simplified from Text-to-LoRA
https://arxiv.org/pdf/2506.06105

Adapter generation follows VeRA (Kopiczko et al., ICLR 2024)
https://arxiv.org/abs/2310.11454
"""

D_INIT = 0.1 # VeRA default init from the paper for d


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
        return x # (B, L*T, head_in_dim)

    def forward(self, x):
        """
        Returns
        -------
        d_mean : (B, L*T, rank)
        b_mean : dict k -> (B, L, d_out[k])
        """

        x = x / 0.8
        h = self._trunk(x)
        d_mean = self.d_head_mean(h) # (B, L*T, rank)
        b_mean = {}
        for k in self.layer_types:
            mask = getattr(self, f"pos_mask_{k}") # (L*T,)
            h_k = h[:, mask, :] # (B, L, head_in)
            b_mean[k] = self.b_head_mean[k](h_k) # (B, L, d_out[k])
        return d_mean, b_mean

    def act(self, x):
        """Deterministic actions (b, d) for the gradient-based trainers.
        Differentiable when called with grad enabled — used for the VJP
        surrogate on the head node."""
        d_mean, b_mean = self.forward(x)
        return b_mean, d_mean
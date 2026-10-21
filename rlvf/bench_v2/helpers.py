from rlvf.model import HyperNetwork
import torch

_STATE_KEYS = ("model_state_dict", "state_dict", "hn_state_dict", "model", "hn")

def _extract_state_dict(obj):
    if isinstance(obj, torch.nn.Module):
        return obj.state_dict()
    if isinstance(obj, dict):
        if "A.q" in obj or any(k.startswith("A.") for k in obj):
            return obj                                    # bare state_dict
        for k in _STATE_KEYS:
            if k in obj:
                return _extract_state_dict(obj[k])
    raise ValueError(
        "could not find a HyperNetwork state_dict in the checkpoint; "
        f"top-level keys: {list(obj.keys()) if isinstance(obj, dict) else type(obj)}")


def _infer_hn_config(sd: dict) -> dict:
    """Recover HyperNetwork constructor args from tensor shapes."""
    layer_types = [k[len("A."):] for k in sd if k.startswith("A.")]
    L, rank, _ = sd[f"A.{layer_types[0]}"].shape
    emb_l = sd["layer_emb.weight"].shape[1]
    emb_t = sd["type_emb.weight"].shape[1]
    context_dim = sd["mixer.layers.1.weight"].shape[1]
    # A HN trained with a whitened input carries a fixed "whiten" buffer
    # (profile_dim, k); the trunk then sees k dims, but the HN still takes
    # raw profile_dim-dimensional profiles.
    whiten = sd.get("whiten")
    if whiten is not None:
        profile_dim = whiten.shape[0]
        assert context_dim == whiten.shape[1] + emb_l + emb_t, \
            (context_dim, whiten.shape, emb_l, emb_t)
    else:
        profile_dim = context_dim - emb_l - emb_t
    return dict(
        num_layers=L,
        layer_types=layer_types,
        dims={k: [sd[f"B.{k}"].shape[1], sd[f"A.{k}"].shape[2]] for k in layer_types},
        emb_l_dim=emb_l,
        emb_t_dim=emb_t,
        profile_dim=profile_dim,
        rank=rank,
        head_in_dim=sd["mlp3.layers.4.weight"].shape[0],
        whiten=whiten,
    )

def load_hn(ckpt_path, device):
    """
    Returns (act, A, B):
      act(profiles (N, 19)) -> (b_batch, d_flat), deterministic (HN means)
          b_batch: {"q": (N, L, d_out_q), "v": (N, L, d_out_v)}
          d_flat:  (N, L*T, rank)
      A, B: the frozen VeRA matrices stored in the HN (its A/B parameters).
    """

    sd = _extract_state_dict(torch.load(ckpt_path, map_location=device,
                                        weights_only=False))
    cfg = _infer_hn_config(sd)
    hn = HyperNetwork(**cfg).to(device)
    result = hn.load_state_dict(sd, strict=True)
    assert not result.missing_keys and not result.unexpected_keys, result
    hn.eval()
    print(f"[HN] {ckpt_path}: L={cfg['num_layers']} types={cfg['layer_types']} "
          f"dims={cfg['dims']} rank={cfg['rank']} head_in={cfg['head_in_dim']} "
          f"input={'whitened ' + str(tuple(cfg['whiten'].shape)) if cfg['whiten'] is not None else 'raw'}")

    A = {k: v.detach() for k, v in hn.A.items()}
    B = {k: v.detach() for k, v in hn.B.items()}
    return hn, A, B

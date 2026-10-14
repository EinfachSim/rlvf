"""
AITA behavioural evaluation across all conditions.

Conditions (each runs on its own GPU worker, all on the SAME profiles & posts):
  prompted   base model, profile rendered into the prompt (no adapter)
  random     random VeRA A/B and random b, d per profile
  warmstart  HN from --ckpt_warmstart
  coldstart  HN from --ckpt_coldstart
  pathwise   HN from --ckpt_pathwise

Usage:
  python aita_eval.py --dataset /path/aita_dataset_10_values_reduced.json \
      --ckpt_warmstart ... --ckpt_coldstart ... --ckpt_pathwise ... \
      [--conditions prompted random pathwise] [--eval_size 64] \
      [--posts_per_side 10] [--out results/aita]

Smoke run (plumbing only):
  python aita_eval.py --dataset ... --conditions prompted random \
      --eval_size 4 --posts_per_side 1
"""
import argparse
import json
import pickle
import time
from pathlib import Path

import ray
import torch

from env import AITAEnv
from workers import MODEL_PATH, TARGET_LAYERS, LAYER_TYPES, RANK

HN_CONDITIONS = ("warmstart", "coldstart", "pathwise")
ALL_CONDITIONS = ("prompted", "random") + HN_CONDITIONS


# ── Hypernetwork loading ──────────────────────────────────────────────────────

# Where your HyperNetwork class lives (override with --hn_module).
HN_MODULE = "rlvf.hypernetwork"

# Keys under which a state_dict may be nested inside a checkpoint dict.
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
    return dict(
        num_layers=L,
        layer_types=layer_types,
        dims={k: [sd[f"B.{k}"].shape[1], sd[f"A.{k}"].shape[2]] for k in layer_types},
        emb_l_dim=emb_l,
        emb_t_dim=emb_t,
        profile_dim=context_dim - emb_l - emb_t,
        rank=rank,
        head_in_dim=sd["mlp3.layers.4.weight"].shape[0],
    )


def load_hypernetwork(ckpt_path: str, device: str, hn_module: str = HN_MODULE):
    """
    Returns (act, A, B):
      act(profiles (N, 19)) -> (b_batch, d_flat), deterministic (HN means)
          b_batch: {"q": (N, L, d_out_q), "v": (N, L, d_out_v)}
          d_flat:  (N, L*T, rank)
      A, B: the frozen VeRA matrices stored in the HN (its A/B parameters).
    """
    import importlib
    HyperNetwork = importlib.import_module(hn_module).HyperNetwork

    sd = _extract_state_dict(torch.load(ckpt_path, map_location=device,
                                        weights_only=False))
    cfg = _infer_hn_config(sd)
    hn = HyperNetwork(**cfg).to(device)
    hn.load_state_dict(sd)                     # strict: catches config mismatch
    hn.eval()
    print(f"[HN] {ckpt_path}: L={cfg['num_layers']} types={cfg['layer_types']} "
          f"dims={cfg['dims']} rank={cfg['rank']} head_in={cfg['head_in_dim']}")

    @torch.no_grad()
    def act(profiles):
        b, d = hn.act(profiles.to(device))
        return b, d

    A = {k: v.detach() for k, v in hn.A.items()}
    B = {k: v.detach() for k, v in hn.B.items()}
    return act, A, B


# ── Random condition ──────────────────────────────────────────────────────────

def model_dims():
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(MODEL_PATH, local_files_only=True)
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    return {
        "L": len(TARGET_LAYERS), "T": len(LAYER_TYPES), "rank": RANK,
        "d_in": cfg.hidden_size,
        "d_out": {"q": cfg.num_attention_heads * head_dim,
                  "v": cfg.num_key_value_heads * head_dim},
    }


def make_random_condition(n: int, seed: int, b_scale: float, d_scale: float,
                          d_mean: float = 0.0):
    """
    Random VeRA: A, B Kaiming-uniform (same init as the HN), b ~ N(0, b_scale²),
    d ~ N(d_mean, d_scale²), independently per profile. Choose the scales so the
    random adapters perturb the model about as much as the HN adapters do
    (compare b_abs_mean / d_abs_mean in the printed diagnostics).
    """
    dims = model_dims()
    L, T, r, d_in, d_out = dims["L"], dims["T"], dims["rank"], dims["d_in"], dims["d_out"]
    g = torch.Generator().manual_seed(seed)

    def kaiming_uniform(shape):
        # Exactly the HyperNetwork's init: torch's kaiming_uniform_ on the 3D
        # tensor (fan_in = size(1) * size(2)), so random A/B have the same
        # scale as the HN's frozen A/B.
        t = torch.empty(shape)
        torch.nn.init.kaiming_uniform_(t, generator=g)
        return t

    A = {t: kaiming_uniform((L, r, d_in)) for t in LAYER_TYPES}
    B = {t: kaiming_uniform((L, d_out[t], r)) for t in LAYER_TYPES}
    b = {t: torch.randn((n, L, d_out[t]), generator=g) * b_scale for t in LAYER_TYPES}
    d = d_mean + torch.randn((n, L * T, r), generator=g) * d_scale
    return (b, d), (A, B)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--conditions", nargs="+", default=list(ALL_CONDITIONS),
                    choices=ALL_CONDITIONS)
    ap.add_argument("--ckpt_warmstart")
    ap.add_argument("--ckpt_coldstart")
    ap.add_argument("--ckpt_pathwise")
    ap.add_argument("--eval_size", type=int, default=64)
    ap.add_argument("--posts_per_side", type=int, default=10,
                    help="per value: this many posts where high-value says YTA, and as many where it says NTA")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--random_b_scale", type=float, default=1.0)
    ap.add_argument("--random_d_scale", type=float, default=0.1)
    ap.add_argument("--random_d_mean", type=float, default=0.0)
    ap.add_argument("--hn_module", default=HN_MODULE,
                    help="python module containing the HyperNetwork class")
    ap.add_argument("--out", default="results/aita")
    args = ap.parse_args()

    for c in args.conditions:
        if c in HN_CONDITIONS and getattr(args, f"ckpt_{c}") is None:
            ap.error(f"condition '{c}' needs --ckpt_{c}")

    ray.init()
    env = AITAEnv(
        dataset_path=args.dataset,
        posts_per_side=args.posts_per_side,
        worker_types=args.conditions,
        eval_size=args.eval_size,
        rng_seed=args.seed,
    )
    profiles = env.get_observation_batch()            # (N, 19), fixed by seed

    # Build each condition's adapters on the driver (HNs are small).
    actions, frozen = {}, {}
    for c in args.conditions:
        if c == "prompted":
            actions[c] = None
        elif c == "random":
            actions[c], frozen[c] = make_random_condition(
                len(profiles), args.seed, args.random_b_scale, args.random_d_scale,
                args.random_d_mean)
        else:
            act, A, B = load_hypernetwork(getattr(args, f"ckpt_{c}"), "cpu",
                                          args.hn_module)
            actions[c] = act(profiles)
            frozen[c] = (A, B)
        if actions[c] is not None:
            b, d = actions[c]
            print(f"[{c}] |b|={float(torch.cat([v.flatten() for v in b.values()]).abs().mean()):.4f} "
                  f"|d|={float(d.abs().mean()):.4f}")

    t0 = time.perf_counter()
    eval_out = env.step_batch(actions, frozen)
    print(f"Benchmark done in {(time.perf_counter() - t0) / 60:.1f} min")

    # Save: summary (json) + everything needed to recompute metrics (pickle).
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    with open(out / f"summary_{stamp}.json", "w") as f:
        json.dump({"args": vars(args), **eval_out}, f, indent=2)
    with open(out / f"raw_{stamp}.pkl", "wb") as f:
        pickle.dump({"args": vars(args), "profiles": profiles.numpy(),
                     "post_ids": env.post_ids, "results": env.last_results}, f)
    print(f"Saved to {out}/summary_{stamp}.json and raw_{stamp}.pkl")


if __name__ == "__main__":
    main()
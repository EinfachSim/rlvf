"""
Evaluation across conditions, on AITA (behaviour; methodology adapted from
Nie et al. 2025, arXiv:2508.11414 -- see aita_metrics.py) or PVQ-RR (the
questionnaire the HNs were trained on).

Conditions (all on the SAME profiles and tasks; any number of GPUs):
  base       plain base model, no conditioning (run once, profile-independent)
  prompted   base model, profile rendered into the prompt (no adapter)
  random     random VeRA A/B and random b, d per profile, scale-matched to an HN
  <name>     any HN checkpoint, given as --hn <name>=<path> (repeatable);
             legacy --ckpt_pathwise / --ckpt_warmstart / --ckpt_coldstart work too
Every conditioned condition also runs once with the eval prior's mean profile
(the average person), the reference for the probability gain.
Re-score a finished run without the GPU:  python recompute.py results/aita/raw_<stamp>.pkl

AITA protocols (--protocols, default both, on the same workers / model load):
  profiles  64 random profiles: alignment r, probability gain, specificity, lean
  s2b       Survey-to-Behavior Table 2 protocol (s2b_eval.py): one value lowered
            (and raised) at a time, gain vs. the base model, next to the paper's numbers

Usage (from the bench folder, rlvf importable):
  python aita_eval.py --benchmark aita --dataset /path/aita_dataset_10_values_reduced.json \
      --hn pathwise=/path/hn_step_00215.pt --seed 1234
  python aita_eval.py --benchmark pvq --hn pathwise=/path/hn_step_00215.pt --seed 1234

  --seed 1234: eval profiles the HNs never saw (training eval used 42 / 43).
Smoke run (plumbing only):
  python aita_eval.py --benchmark aita --dataset ... --conditions base prompted \
      --eval_size 4 --posts_per_side 2 --n_boot 200
"""
import argparse
import json
import pickle
import time
from pathlib import Path

import ray
import torch

from env import AITAEnv, PVQEnv
from workers import MODEL_PATH, TARGET_LAYERS, LAYER_TYPES, RANK

FIXED_CONDITIONS = ("base", "prompted", "random")
LEGACY_HN = ("warmstart", "coldstart", "pathwise")


# ── Hypernetwork loading ──────────────────────────────────────────────────────

# Where your HyperNetwork class lives (override with --hn_module).
HN_MODULE = "rlvf.model"

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
    result = hn.load_state_dict(sd, strict=True)             # strict: catches config mismatch
    assert not result.missing_keys and not result.unexpected_keys, result
    hn.eval()
    print(f"[HN] {ckpt_path}: L={cfg['num_layers']} types={cfg['layer_types']} "
          f"dims={cfg['dims']} rank={cfg['rank']} head_in={cfg['head_in_dim']} "
          f"input={'whitened ' + str(tuple(cfg['whiten'].shape)) if cfg['whiten'] is not None else 'raw'}")

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
    ap.add_argument("--benchmark", choices=["aita", "pvq"], default="aita")
    ap.add_argument("--dataset", help="AITA json (needed for --benchmark aita)")
    ap.add_argument("--quest_path", default=None,
                    help="PVQ-RR questionnaire.txt (default: training path)")
    ap.add_argument("--hn", action="append", default=[], metavar="NAME=PATH",
                    help="HN checkpoint condition (repeatable)")
    for c in LEGACY_HN:
        ap.add_argument(f"--ckpt_{c}", help=f"same as --hn {c}=PATH")
    ap.add_argument("--conditions", nargs="+", default=None,
                    help="default: base prompted random + every --hn")
    ap.add_argument("--num_workers", type=int, default=None,
                    help="GPU workers (default: all GPUs, at most one per condition)")
    ap.add_argument("--eval_size", type=int, default=64)
    ap.add_argument("--posts_per_side", type=int, default=25,
                    help="per value: this many posts where the low standard is the harsher "
                         "verdict, and as many where it is the milder one")
    ap.add_argument("--verdicts", choices=["ternary", "binary"], default="ternary",
                    help="ternary = YTA/NTA/Neutral as in Nie et al. 2025")
    ap.add_argument("--scoring", choices=["sequence", "first_token"], default="first_token",
                    help="first_token = the paper's next-token probability (validated against "
                         "sequence scoring: r = 1.00 on the pilot); sequence = full-verdict "
                         "log-likelihood")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--compare_to", default="prompted",
                    help="paired mean_r differences of every condition against this one")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--eval_means", choices=["empirical", "zero"], default="empirical",
                    help="profile prior for the eval set: 'empirical' = population means "
                         "(Schwartz & Cieciuch 2022, Table 5), 'zero' = uniform coverage")
    ap.add_argument("--random_match", default=None,
                    help="HN condition whose b/d statistics the random condition copies "
                         "(default: the first HN); 'none' = use the scales below")
    ap.add_argument("--random_b_scale", type=float, default=1.0)
    ap.add_argument("--random_d_scale", type=float, default=0.1)
    ap.add_argument("--random_d_mean", type=float, default=0.0)
    ap.add_argument("--protocols", nargs="+", choices=["profiles", "s2b"],
                    default=["profiles", "s2b"], help="AITA only; both share one model load")
    ap.add_argument("--shift_sd", type=float, nargs="+", default=[2.0, 4.0],
                    help="s2b: how far the target value is moved, in SD")
    ap.add_argument("--directions", nargs="+", choices=["low", "high"], default=["low", "high"],
                    help="s2b: lower (paper) and/or raise the target value")
    ap.add_argument("--s2b_posts", choices=["target", "all"], default="target",
                    help="s2b: all = also measure off-target change (~10x cost)")
    ap.add_argument("--s2b_reference", choices=["base", "average"], default="base",
                    help="s2b: base = untuned model (the paper's comparison)")
    ap.add_argument("--hn_module", default=HN_MODULE,
                    help="python module containing the HyperNetwork class")
    ap.add_argument("--out", default=None,
                    help="output dir (default: results/<benchmark>)")
    args = ap.parse_args()

    hns = {}
    for item in args.hn:
        name, path = item.split("=", 1)
        assert name not in FIXED_CONDITIONS, f"HN name {name!r} is reserved"
        hns[name] = path
    for c in LEGACY_HN:
        if getattr(args, f"ckpt_{c}"):
            hns[c] = getattr(args, f"ckpt_{c}")
    conditions = args.conditions or (list(FIXED_CONDITIONS) + list(hns))
    for c in conditions:
        if c not in FIXED_CONDITIONS and c not in hns:
            ap.error(f"condition {c!r} needs --hn {c}=PATH")
    if args.benchmark == "aita" and not args.dataset:
        ap.error("--benchmark aita needs --dataset")
    args.out = args.out or f"results/{args.benchmark}"

    ray.init()
    hn_acts = {c: load_hypernetwork(hns[c], "cpu", args.hn_module) for c in conditions if c in hns}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    meta = {"args": vars(args), "conditions": conditions, "hn_checkpoints": hns}
    protocols = args.protocols if args.benchmark == "aita" else ["profiles"]

    common = dict(eval_size=args.eval_size, rng_seed=args.seed, num_workers=args.num_workers,
                  eval_means=None if args.eval_means == "zero" else "empirical")
    workers = None
    if "profiles" in protocols:
        if args.benchmark == "aita":
            env = AITAEnv(dataset_path=args.dataset, posts_per_side=args.posts_per_side,
                          verdicts=args.verdicts, scoring=args.scoring, n_boot=args.n_boot,
                          compare_to=args.compare_to, worker_types=conditions, **common)
        else:
            env = PVQEnv(quest_path=args.quest_path, worker_types=conditions, **common)
        workers = env.workers
        run_profiles(env, conditions, hn_acts, args, out, stamp, meta)

    if "s2b" in protocols:
        from s2b_eval import S2BEnv
        s2b_conds = [c for c in conditions if c != "random"]      # random adapters: no target
        if args.s2b_reference == "base" and "base" not in s2b_conds:
            s2b_conds = ["base"] + s2b_conds
        env = S2BEnv(args.dataset, shift_sd=args.shift_sd, directions=args.directions,
                     posts=args.s2b_posts, verdicts=args.verdicts, scoring=args.scoring,
                     reference=args.s2b_reference, worker_types=s2b_conds,
                     workers=workers, **{k: v for k, v in common.items() if k != "eval_size"})
        actions, frozen = {}, {}
        for c in s2b_conds:
            if c in hn_acts:
                act, A, B = hn_acts[c]
                actions[c], frozen[c] = act(env.action_profiles), (A, B)
            else:
                actions[c] = None
        t0 = time.perf_counter()
        summary = env.step_batch(actions, frozen)
        print(f"S2B protocol done in {(time.perf_counter() - t0) / 60:.1f} min")
        s2b_meta = {**meta, "conditions": s2b_conds, "targets": env.targets}
        with open(out / f"s2b_summary_{stamp}.json", "w") as f:
            json.dump({**s2b_meta, **summary}, f, indent=2, default=float)
        with open(out / f"s2b_raw_{stamp}.pkl", "wb") as f:
            pickle.dump({**s2b_meta, "profiles": env.action_profiles.numpy(),
                         "results": env.last_results}, f)
        print(f"Saved to {out}/s2b_summary_{stamp}.json and s2b_raw_{stamp}.pkl")


def run_profiles(env, conditions, hn_acts, args, out, stamp, meta):
    profiles = env.action_profiles            # (N + 1, 19): eval profiles + reference
    # Build each condition's adapters on the driver (HNs are small).
    actions, frozen = {}, {}
    for c in conditions:
        if c in hn_acts:
            act, A, B = hn_acts[c]
            actions[c], frozen[c] = act(profiles), (A, B)
        else:
            actions[c] = None
    if "random" in conditions:
        match = args.random_match or next(iter(hn_acts), "none")
        if match != "none":
            b, d = actions[match]
            b_all = torch.cat([v.flatten() for v in b.values()])
            args.random_b_scale, args.random_d_mean, args.random_d_scale = \
                float(b_all.std()), float(d.mean()), float(d.std())
            print(f"[random] matched to {match}: b~N(0,{args.random_b_scale:.3f}^2), "
                  f"d~N({args.random_d_mean:.3f},{args.random_d_scale:.3f}^2)")
        actions["random"], frozen["random"] = make_random_condition(
            len(profiles), args.seed, args.random_b_scale, args.random_d_scale,
            args.random_d_mean)
    for c in conditions:
        if actions[c] is not None:
            b, d = actions[c]
            print(f"[{c}] |b|={float(torch.cat([v.flatten() for v in b.values()]).abs().mean()):.4f} "
                  f"|d|={float(d.abs().mean()):.4f}")

    t0 = time.perf_counter()
    eval_out = env.step_batch(actions, frozen)
    print(f"Benchmark done in {(time.perf_counter() - t0) / 60:.1f} min")

    # Save: summary (json) + everything needed to recompute metrics (pickle).
    with open(out / f"summary_{stamp}.json", "w") as f:
        json.dump({**meta, **eval_out}, f, indent=2, default=float)
    with open(out / f"raw_{stamp}.pkl", "wb") as f:
        pickle.dump({**meta, "profiles": profiles.numpy(),
                     "post_ids": env.post_ids, "results": env.last_results}, f)
    print(f"Saved to {out}/summary_{stamp}.json and raw_{stamp}.pkl")


if __name__ == "__main__":
    main()
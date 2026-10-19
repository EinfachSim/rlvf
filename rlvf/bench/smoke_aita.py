"""
AITA smoke test — single GPU, no Ray.

Checks, in order:
  1. Candidate tokenization: how " YTA" / " NTA" tokenize (length bias check).
  2. Base model: p_yta distribution on a small balanced sample — must not be
     degenerate (all ~0.5, or all one verdict).
  3. Prompted baseline, directional: for one value, a strongly-high vs
     strongly-low profile should shift verdicts toward high_stance / low_stance.
  4. (optional, --adapter) the same directional check with a saved HN adapter.

Usage:
  python smoke_aita.py --dataset /path/aita_dataset_10_values_reduced.json \
      [--n_per_value 3] [--value Tradition] [--adapter adapter.pt]

adapter.pt = torch.save({"b": {"q":..,"v":..}, "d":.., "A": {..}, "B": {..},
                         "profile": [19 floats]}, ...)
"""
import argparse
import random
import statistics
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from runners import AITARunner

MODEL_PATH = "/lustre/mlnvme/data/s03skoeh_hpc-rlvf/models/Mistral-7B-Instruct-v0.3"

COARSE_TO_REFINED = {
    "Self_direction": [0, 1], "Stimulation": [2], "Hedonism": [3],
    "Achievement": [4], "Power": [5, 6], "Security": [8, 9],
    "Tradition": [10], "Conformity": [11, 12],
    "Universalism": [14, 15, 16], "Benevolence": [17, 18],
}


# ── helpers ───────────────────────────────────────────────────────────────────

def extreme_profile(value: str, sign: float, magnitude: float = 2.0) -> list[float]:
    """Ipsatized 19-dim profile: `value` dims at ±magnitude, rest equal, mean 0."""
    p = torch.zeros(19)
    idx = COARSE_TO_REFINED[value]
    p[idx] = sign * magnitude
    p -= p.mean()
    return p.tolist()


def apply_vera(model, b, d, A, B, device, stats: dict | None = None):
    """Same hook math as BaseWorker._apply_vera. Returns hook handles.
    If `stats` is a dict, records ||delta|| / ||W0 x|| per (layer, type)."""
    hooks = []
    for li in range(32):
        attn = model.model.layers[li].self_attn
        for ti, t in enumerate(["q", "v"]):
            d_lt = d[li, ti].to(device, torch.float32)
            b_lt = b[t][li].to(device, torch.float32)
            A_lt = A[t][li].to(device, torch.float32)
            B_lt = B[t][li].to(device, torch.float32)
            proj = attn.q_proj if t == "q" else attn.v_proj

            def hook(module, inputs, output, A_lt=A_lt, B_lt=B_lt,
                     d_lt=d_lt, b_lt=b_lt, key=(li, t)):
                x = inputs[0].to(torch.float32)
                delta = (((x @ A_lt.T) * d_lt) @ B_lt.T) * b_lt
                if stats is not None:
                    stats.setdefault(key, []).append(
                        float(delta.norm() / output.float().norm().clamp_min(1e-12)))
                return output + delta.to(output.dtype)

            hooks.append(proj.register_forward_hook(hook))
    return hooks


def make_random_adapter(model, seed: int, b_scale: float, d_scale: float,
                        ab_path: str | None = None):
    """
    Random VeRA adapter for the 'random' condition / smoke tests.

    A, B: if `ab_path` is given, load the frozen matrices used in HN training
    (strongly preferred: then 'random' differs from the HN conditions only in
    b, d). Otherwise generate them like VeRA does (Kaiming-uniform, seeded).
    b, d: i.i.d. Gaussian with the given std. Pick b_scale/d_scale to match the
    b_abs_mean / d_abs_mean your HN produces (logged by run_episode), so the
    random adapter perturbs the model by a comparable amount.
    """
    cfg = model.config
    L, r = cfg.num_hidden_layers, 8
    d_in = cfg.hidden_size
    head_dim = getattr(cfg, "head_dim", None) or d_in // cfg.num_attention_heads
    d_out = {"q": cfg.num_attention_heads * head_dim,
             "v": cfg.num_key_value_heads * head_dim}
    g = torch.Generator().manual_seed(seed)

    if ab_path:
        ab = torch.load(ab_path, map_location="cpu")
        A, B = ab["A"], ab["B"]
    else:
        def kaiming_uniform(shape):          # fan_in = shape[-1]
            bound = (6.0 / shape[-1]) ** 0.5
            return (torch.rand(shape, generator=g) * 2 - 1) * bound
        A = {t: kaiming_uniform((L, r, d_in)) for t in ("q", "v")}
        B = {t: kaiming_uniform((L, d_out[t], r)) for t in ("q", "v")}

    b = {t: torch.randn((L, d_out[t]), generator=g) * b_scale for t in ("q", "v")}
    d = torch.randn((L, 2, r), generator=g) * d_scale
    print(f"  A/B from {'file ' + ab_path if ab_path else 'seeded Kaiming init'}; "
          f"b~N(0,{b_scale}^2) |b|={float(torch.cat([v.flatten() for v in b.values()]).abs().mean()):.4f}, "
          f"d~N(0,{d_scale}^2) |d|={float(d.abs().mean()):.4f}")
    return {"b": b, "d": d, "A": A, "B": B}


def summarize(tag, posts):
    ps = [p["p_yta"] for p in posts]
    n_yta = sum(p["verdict"] == "YTA" for p in posts)
    print(f"  [{tag}] n={len(ps)}  p_yta mean={statistics.mean(ps):.3f} "
          f"sd={statistics.pstdev(ps):.3f} min={min(ps):.3f} max={max(ps):.3f}  "
          f"YTA={n_yta}/{len(ps)}")


def agreement(posts, stance_key):
    """Fraction of scoreable (non-Neutral) posts where verdict == expected."""
    s = [p for p in posts if p[stance_key] in ("YTA", "NTA")]
    if not s:
        return float("nan"), 0
    return sum(p["verdict"] == p[stance_key] for p in s) / len(s), len(s)


def directional_check(tag, runner, model, tok, device, value, post_ids,
                      system_prompt_fn=None):
    hi = runner.run(model, tok, device, {"post_ids": post_ids},
                    system_prompt=system_prompt_fn(+1) if system_prompt_fn else None)["posts"]
    lo = runner.run(model, tok, device, {"post_ids": post_ids},
                    system_prompt=system_prompt_fn(-1) if system_prompt_fn else None)["posts"]
    a_hi, n_hi = agreement(hi, "high_stance")
    a_lo, n_lo = agreement(lo, "low_stance")
    dp = statistics.mean(h["p_yta"] - l["p_yta"] for h, l in zip(hi, lo))
    print(f"  [{tag}] high profile agrees with high_stance: {a_hi:.2f} (n={n_hi})")
    print(f"  [{tag}] low  profile agrees with low_stance : {a_lo:.2f} (n={n_lo})")
    print(f"  [{tag}] mean |Δp_yta| high-vs-low: "
          f"{statistics.mean(abs(h['p_yta'] - l['p_yta']) for h, l in zip(hi, lo)):.3f} "
          f"(signed {dp:+.3f})")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", default=MODEL_PATH)
    ap.add_argument("--n_per_value", type=int, default=3)
    ap.add_argument("--value", default="Tradition",
                    help="value for the directional check")
    ap.add_argument("--n_directional", type=int, default=20)
    ap.add_argument("--adapter", default=None, help="saved HN adapter .pt")
    ap.add_argument("--random_adapter", action="store_true",
                    help="use a random VeRA adapter instead of --adapter")
    ap.add_argument("--ab_path", default=None,
                    help=".pt with the frozen {'A','B'} used in HN training")
    ap.add_argument("--b_scale", type=float, default=1.0)
    ap.add_argument("--d_scale", type=float, default=0.1)
    ap.add_argument("--adapter_seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda:0"
    rng = random.Random(args.seed)

    print("Loading model...")
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, local_files_only=True).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    print(f"  loaded in {time.perf_counter() - t0:.1f}s")

    runner = AITARunner(args.dataset, batch_size=8)

    # 1. tokenization
    print("\n[1] Candidate tokenization")
    for c in runner.VERDICTS:
        ids = tok(c, add_special_tokens=False).input_ids
        print(f"  {c!r} -> {ids} {tok.convert_ids_to_tokens(ids)}")
    lens = {len(tok(c, add_special_tokens=False).input_ids) for c in runner.VERDICTS}
    if len(lens) > 1:
        print("  WARNING: candidates differ in token count -> summed log-lik is "
              "length-biased; consider comparing first-token logprobs only.")
    from runners import build_chat_ids
    demo = build_chat_ids(tok, runner.instruction.format(post="<POST>"), None, 2048)
    print("  prompt tail:", repr(tok.decode(demo[-30:])))

    # 2. base model distribution
    print("\n[2] Base model, balanced sample")
    by_value = {}
    for it in runner.items:
        by_value.setdefault(it["value"], []).append(it["post_id"])
    sample = [pid for ids in by_value.values()
              for pid in rng.sample(ids, min(args.n_per_value, len(ids)))]
    t0 = time.perf_counter()
    base = runner.run(model, tok, device, {"post_ids": sample})["posts"]
    dt = time.perf_counter() - t0
    summarize("base", base)
    print(f"  {dt / len(sample):.2f}s per post "
          f"(→ ~{dt / len(sample) * 200 * 64 / 3600:.1f} GPU-h for 64 profiles × 200 posts)")
    for p in base[:5]:
        print(f"    {p['post_id']:20s} p_yta={p['p_yta']:.3f} "
              f"high={p['high_stance']} low={p['low_stance']}")

    # 3. prompted directional
    print(f"\n[3] Prompted baseline, directional on {args.value}")
    from workers import render_profile_prompt
    dir_ids = rng.sample(by_value[args.value],
                         min(args.n_directional, len(by_value[args.value])))
    directional_check(
        "prompted", runner, model, tok, device, args.value, dir_ids,
        system_prompt_fn=lambda s: render_profile_prompt(extreme_profile(args.value, s)),
    )
    print("  Example prompt (high):")
    print("   ", render_profile_prompt(extreme_profile(args.value, +1))[:300], "...")

    # 4. adapter
    if args.adapter or args.random_adapter:
        if args.random_adapter:
            print(f"\n[4] Random adapter (seed={args.adapter_seed})")
            ad = make_random_adapter(model, args.adapter_seed,
                                     args.b_scale, args.d_scale, args.ab_path)
        else:
            print(f"\n[4] Adapter {args.adapter}")
            ad = torch.load(args.adapter, map_location="cpu")

        # Perturbation size: ||ΔW x|| / ||W0 x|| on one short forward.
        stats = {}
        hooks = apply_vera(model, ad["b"], ad["d"], ad["A"], ad["B"], device, stats)
        try:
            with torch.no_grad():
                probe = tok("The quick brown fox jumps over the lazy dog.",
                            return_tensors="pt").to(device)
                model(**probe)
        finally:
            for h in hooks:
                h.remove()
        for t in ("q", "v"):
            r = [v[0] for (li, tt), v in stats.items() if tt == t]
            print(f"  ||ΔWx||/||W0x|| [{t}]: mean={statistics.mean(r):.4f} "
                  f"max={max(r):.4f}")
        if any(not torch.isfinite(torch.tensor(v[0])) for v in stats.values()):
            print("  WARNING: non-finite delta — fp16 overflow, lower the scales")

        hooks = apply_vera(model, ad["b"], ad["d"], ad["A"], ad["B"], device)
        try:
            adapted = runner.run(model, tok, device, {"post_ids": sample})["posts"]
        finally:
            for h in hooks:
                h.remove()
        summarize("adapted", adapted)
        shift = statistics.mean(abs(a["p_yta"] - b["p_yta"])
                                for a, b in zip(adapted, base))
        print(f"  mean |Δp_yta| vs base: {shift:.3f}")
        if "profile" in ad:
            prof = torch.tensor(ad["profile"])
            print("  per-value: profile score, agreement with expected stance")
            for v, idx in COARSE_TO_REFINED.items():
                s = float(prof[idx].mean())
                key = "high_stance" if s > 0 else "low_stance"
                posts_v = [p for p in adapted if p["value"] == v]
                acc, n = agreement(posts_v, key)
                print(f"    {v:15s} score={s:+.2f}  expects {key:11s} "
                      f"agree={acc:.2f} (n={n})")

    print("\nDone.")


if __name__ == "__main__":
    main()
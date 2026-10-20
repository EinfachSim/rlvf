"""
Survey-to-Behavior replication protocol (Nie et al. 2025, arXiv:2508.11414,
Table 2), so AITA results can be compared with the paper directly.

Their experiment: fine-tune a model so that ONE value is rated low on the
survey, everything else unchanged; then measure on that value's AITA posts the
PROBABILITY GAIN against the unmanipulated control model:
    gain = [P_ref(high stance) - P_X(high stance)] + [P_X(low stance) - P_ref(low stance)]
         = -(pi_X - pi_ref),          pi = P(high stance) - P(low stance)
with P over {YTA, NTA, Neutral}, renormalized, next-token probabilities;
averaged over the value's posts; "Weighted Average" over values weighted by
the number of posts (this reproduces their reported Qwen3 11.4 % and Falcon3
0.87 %).

Same protocol here, with conditioning instead of fine-tuning:
  reference (--reference):
     base     = the plain base model, no conditioning. This is the paper's
                comparison (confirmed by the authors: gains are measured
                against the model before tuning) [default]
     average  = the same condition with the AVERAGE-PERSON profile
                (ipsatized population means): removes the condition's
                generic offset, i.e. isolates the effect of lowering X
  Both are always computed; --reference picks the one in the main columns.
  lowered X = the same profile with all of X's PVQ-RR dimensions lowered by
              k SD (re-ipsatized). k = 2 stays inside the training
              distribution; k = 4 is about -3 answer points relative to the
              person's mean, i.e. roughly their "rating 1" (approximate:
              exact equivalence depends on the person's overall mean).
  Optional: raised X (--directions low high), not in the paper.
Each targeted episode judges ALL posts of its value (as in the paper; no
side balancing). The reference episode judges every selected post.
With --posts all, targeted episodes judge every post, which adds the paper's
other analysis: change on OTHER values' posts (off-target, mean |delta pi|).

Lean control: as in the paper, a model that simply says YTA more often scores a
positive gain on every post whose low-standard stance is the harsher verdict
(most posts for several values, e.g. Self_direction 475/500). The "bal"
column therefore also reports the side-balanced gain (posts where the low
stance is harsher and posts where it is milder weighted equally); a real
value effect survives balancing, a pure lean shift averages out. Running
--directions low high gives a second check: lean shifts flip sign, value
effects do not.

Comparability caveats (state them in the thesis):
  * base model differs (Mistral-7B-Instruct-v0.3 vs LLaMA3.1 / Qwen3 / Falcon3)
  * the 10-value reduced dataset groups the paper's 20 labels; the paper's
    per-value numbers are aggregated to these 10 groups by sample count
  * their AITA prompt is not published; ours is in runners.py
  * strength of "rating 1" vs a k-SD shift is only approximately matched

Usage (bench folder):
  python s2b_eval.py --dataset .../aita_dataset_10_values_reduced.json \\
      --hn pathwise=.../hn_step_00235.pt --conditions base prompted pathwise \\
      --scoring first_token --shift_sd 2 4
"""
import argparse
import json
import pickle
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import ray
import torch

import aita_metrics as AM
from aita_eval import load_hypernetwork, HN_MODULE
from env import BaseEnv
from rlvf.data import PROFILE_SD

# Paper Table 2, aggregated to the dataset's 10 coarse values by sample count
# (LLaMA3.1-8B, Qwen3-8B, Falcon3-7B; percentage points; n = their sample count).
PAPER_TABLE2 = {
    "Self_direction": (526, 3.7, 14.8, 2.6),
    "Stimulation":    (13, 3.0, 13.4, 4.9),
    "Hedonism":       (20, -48.7, 24.9, -2.8),
    "Achievement":    (95, 3.7, 25.1, 1.4),
    "Power":          (28, 2.5, 8.8, 1.8),
    "Security":       (510, 5.3, 33.9, -6.2),
    "Tradition":      (31, -1.0, -6.8, 2.5),
    "Conformity":     (833, -0.9, -4.5, -0.3),
    "Universalism":   (1001, 1.4, 14.2, 4.7),
    "Benevolence":    (1000, 4.8, 10.8, 1.9),
}
PAPER_WAVG = {"LLaMA3.1-8B": 2.89, "Qwen3-8B": 11.4, "Falcon3-7B": 0.87}   # as reported
PAPER_MODELS = list(PAPER_WAVG)


class S2BEnv(BaseEnv):
    BROADCAST_BASE = True
    USE_REFERENCE = True

    def __init__(self, dataset_path, shift_sd=(2.0, 4.0), directions=("low",),
                 posts="target", max_posts_per_value=None, verdicts="ternary",
                 scoring="first_token", reference="base", **base_kwargs):
        super().__init__(benchmark="aita",
                         runner_kwargs={"dataset_path": dataset_path, "verdicts": verdicts,
                                        "scoring": scoring},
                         eval_size=1, **base_kwargs)
        with open(dataset_path) as f:
            raw = json.load(f)
        rng = random.Random(1234)
        self.posts_by_value = {}
        for value, items in raw.items():
            if value not in AM.VALUE_TO_INDICES:
                continue
            ids = [f"{value}/{j}" for j, it in enumerate(items)
                   if AM._usable(it, verdicts == "ternary")]
            if max_posts_per_value and len(ids) > max_posts_per_value:
                ids = sorted(rng.sample(ids, max_posts_per_value))
            if ids:
                self.posts_by_value[value] = ids
        self.all_posts = sorted(p for ids in self.posts_by_value.values() for p in ids)
        self.post_ids = self.all_posts
        self.posts_mode = posts
        self.reference = reference

        m = self.ref_profile[0]
        self.targets, profs = [], []
        for value in self.posts_by_value:
            for k in shift_sd:
                for d in directions:
                    p = m.clone()
                    p[AM.VALUE_TO_INDICES[value]] += (-1 if d == "low" else 1) * k * PROFILE_SD
                    profs.append(p - p.mean())
                    self.targets.append({"value": value, "shift_sd": k, "direction": d})
        self.eval_profiles = torch.stack(profs)
        print(f"[S2BEnv] {len(self.targets)} targeted profiles "
              f"({len(self.posts_by_value)} values x shifts {list(shift_sd)} x {list(directions)}), "
              f"{len(self.all_posts)} posts; targeted episodes judge "
              f"{'their own value' if posts == 'target' else 'all'} posts")

    def _build_task(self, i):
        if i >= len(self.targets) or self.posts_mode == "all":
            return {"post_ids": self.all_posts}
        return {"post_ids": self.posts_by_value[self.targets[i]["value"]]}

    def _log(self, results, tag="BENCH"):
        by = defaultdict(list)
        for r in results:
            by[r["worker_type"]].append(r)
        for wt, rs in by.items():
            ok = [r for r in rs if r["ok"]]
            print(f"[{tag}] {wt}: {len(ok)}/{len(rs)} episodes ok, "
                  f"{sum(r['metrics']['n_posts'] for r in ok)} verdicts")

    def evaluate_benchmark(self, results):
        summary = evaluate(results, self.targets, n_boot=1000, reference=self.reference)
        print_report(summary)
        return {"benchmark": "s2b", **summary}


# ── Metrics ───────────────────────────────────────────────────────────────────

def _pi_by_post(posts):
    out = {}
    for p in posts:
        pr = AM._probs(p)
        out[p["post_id"]] = (p["value"], pr.get(p["high_stance"], 0.0) - pr.get(p["low_stance"], 0.0),
                             AM.direction(p["high_stance"], p["low_stance"]))
    return out


def _balanced(g, dirs):
    a, b = g[dirs == 1], g[dirs == -1]
    return 0.5 * a.mean() + 0.5 * b.mean() if len(a) and len(b) else float("nan")


def evaluate(results, targets, n_boot=1000, seed=0, reference="base"):
    rng = np.random.default_rng(seed)
    ref, base, eps = {}, None, defaultdict(list)
    for r in results:
        if not r["ok"] or r.get("broadcast"):
            continue
        pis = _pi_by_post(r["metrics"]["posts"])
        if r.get("is_ref"):
            ref[r["worker_type"]] = pis
            if r["worker_type"] == "base":
                base = pis
        else:
            eps[r["worker_type"]].append((r["adapter_id"], pis))
    if reference == "base" and base is None:
        print("[S2B] no base condition in this run -> using --reference average")
        reference = "average"

    out = {}
    for cond, rows in eps.items():
        refs = {"average": ref[cond], "base": base}
        cells = defaultdict(dict)                 # (shift, dir) -> value -> stats
        for aid, pis in rows:
            t = targets[aid]
            sign = -1.0 if t["direction"] == "low" else 1.0
            tgt = [(pid, pi, dr) for pid, (v, pi, dr) in pis.items() if v == t["value"]]
            dirs = np.array([dr for _, _, dr in tgt])
            st = {"n": len(tgt)}
            for name, R in refs.items():
                if R is None:
                    continue
                g = sign * np.array([pi - R[pid][1] for pid, pi, _ in tgt])
                st[f"gain_{name}"] = float(g.mean())
                st[f"gain_{name}_bal"] = float(_balanced(g, dirs))
                if name == reference:
                    st["boot"] = [g[rng.integers(0, len(g), len(g))].mean() for _ in range(n_boot)]
                    st["abs_change_target"] = float(np.abs(g).mean())
                    off = [abs(pi - R[pid][1]) for pid, (v, pi, _) in pis.items() if v != t["value"]]
                    st["abs_change_other"] = float(np.mean(off)) if off else float("nan")
            cells[(t["shift_sd"], t["direction"])][t["value"]] = st

        out[cond] = {}
        for key, vals in cells.items():
            n = np.array([v["n"] for v in vals.values()], float)
            wavg = lambda k: float(np.nansum(n * np.array([v.get(k, np.nan) for v in vals.values()]))
                                   / n[np.isfinite([v.get(k, np.nan) for v in vals.values()])].sum())
            boots = np.array([v.pop("boot") for v in vals.values()])
            wboot = (n[:, None] * boots).sum(0) / n.sum()
            other = [v["abs_change_other"] for v in vals.values()]
            rec = {"reference": reference, "per_value": vals,
                   "gain_wavg": wavg(f"gain_{reference}"),
                   "gain_wavg_ci": [float(np.percentile(wboot, 2.5)), float(np.percentile(wboot, 97.5))],
                   "gain_bal_wavg": wavg(f"gain_{reference}_bal"),
                   "abs_change_target_wavg": wavg("abs_change_target"),
                   "abs_change_other_mean": float(np.nanmean(other)) if np.isfinite(other).any() else float("nan")}
            for name in ("base", "average"):
                if any(f"gain_{name}" in v for v in vals.values()):
                    rec[f"gain_{name}_wavg"] = wavg(f"gain_{name}")
                    rec[f"gain_{name}_bal_wavg"] = wavg(f"gain_{name}_bal")
            out[cond][f"{key[1]}_{key[0]:g}sd"] = rec
    return {"reference": reference, "per_condition": out,
            "paper_table2_coarse": PAPER_TABLE2, "paper_wavg": PAPER_WAVG}


def print_report(summary):
    pc, ref = summary["per_condition"], summary["reference"]
    other_ref = "average" if ref == "base" else "base"
    values = list(PAPER_TABLE2)
    head = " ".join(f"{v[:6]:>7s}" for v in values)
    print(f"\n[S2B] probability gain toward the expected stance, percentage points "
          f"(Nie et al. 2025, Table 2 protocol), vs. {ref.upper()}"
          f"{' (= paper)' if ref == 'base' else ''}")
    print(f"{'condition':22s} {'wavg':>6s} {'95% CI':>15s} {'bal':>6s} "
          f"{'vs_' + other_ref[:3]:>7s} {'|d|tgt':>6s} {'|d|oth':>6s} " + head)
    for cond in sorted(pc):
        for setting, s in sorted(pc[cond].items()):
            ci = s["gain_wavg_ci"]
            cells = " ".join(f"{100 * s['per_value'][v][f'gain_{ref}']:+7.1f}" if v in s["per_value"]
                             else f"{'-':>7s}" for v in values)
            oth, alt = s["abs_change_other_mean"], s.get(f"gain_{other_ref}_wavg", float("nan"))
            print(f"{(cond + ' ' + setting):22s} {100 * s['gain_wavg']:+6.1f} "
                  f"[{100 * ci[0]:+5.1f},{100 * ci[1]:+5.1f}] {100 * s['gain_bal_wavg']:+6.1f} "
                  f"{100 * alt:+7.1f} {100 * s['abs_change_target_wavg']:6.1f} "
                  f"{(f'{100 * oth:6.1f}' if oth == oth else '     -')} " + cells)
    print("  paper (fine-tuning one value to a low survey rating, vs. the model before tuning),"
          " aggregated to these 10 values:")
    for mi, m in enumerate(PAPER_MODELS):
        cells = " ".join(f"{PAPER_TABLE2[v][1 + mi]:+7.1f}" for v in values)
        print(f"{('paper ' + m):22s} {PAPER_WAVG[m]:+6.1f} {'':15s} {'':6s} {'':7s} {'':6s} {'':6s} " + cells)
    any_rec = next((s for c in pc.values() for s in c.values()), {"per_value": {}})
    print("  n posts here:  " + " ".join(f"{v[:6]}={any_rec['per_value'].get(v, {}).get('n', 0)}" for v in values))
    print("  n posts paper: " + " ".join(f"{v[:6]}={PAPER_TABLE2[v][0]}" for v in values))
    print("  wavg: paper metric (all posts, weighted by #posts). bal: same, side-balanced "
          "(lean-controlled). vs_xxx: wavg against the other reference.\n"
          "  |d|tgt / |d|oth: mean |change of pi| on the target value's / other values' posts "
          "(the latter only with --posts all)\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--hn", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--conditions", nargs="+", default=None,
                    help="default: base prompted + every --hn")
    ap.add_argument("--shift_sd", type=float, nargs="+", default=[2.0, 4.0])
    ap.add_argument("--directions", nargs="+", choices=["low", "high"], default=["low"])
    ap.add_argument("--posts", choices=["target", "all"], default="target",
                    help="all = targeted episodes judge every post (adds off-target change; ~10x cost)")
    ap.add_argument("--max_posts_per_value", type=int, default=None,
                    help="cap posts per value (default: all, as in the paper)")
    ap.add_argument("--scoring", choices=["sequence", "first_token"], default="first_token")
    ap.add_argument("--reference", choices=["base", "average"], default="base",
                    help="base = model without conditioning (the paper's comparison); "
                         "average = same condition with the average-person profile")
    ap.add_argument("--eval_means", choices=["empirical", "zero"], default="empirical",
                    help="prior whose mean profile is the reference (average person)")
    ap.add_argument("--num_workers", type=int, default=None)
    ap.add_argument("--hn_module", default=HN_MODULE)
    ap.add_argument("--out", default="results/s2b")
    args = ap.parse_args()

    hns = dict(item.split("=", 1) for item in args.hn)
    conditions = args.conditions or (["base", "prompted"] + list(hns))
    if args.reference == "base" and "base" not in conditions:
        conditions = ["base"] + conditions             # one episode; needed as the reference
    for c in conditions:
        if c not in ("base", "prompted") and c not in hns:
            ap.error(f"condition {c!r} needs --hn {c}=PATH")

    ray.init()
    env = S2BEnv(args.dataset, shift_sd=args.shift_sd, directions=args.directions,
                 posts=args.posts, max_posts_per_value=args.max_posts_per_value,
                 scoring=args.scoring, reference=args.reference,
                 worker_types=conditions, num_workers=args.num_workers,
                 eval_means=None if args.eval_means == "zero" else "empirical")
    profiles = env.action_profiles
    actions, frozen = {}, {}
    for c in conditions:
        if c in hns:
            act, A, B = load_hypernetwork(hns[c], "cpu", args.hn_module)
            actions[c], frozen[c] = act(profiles), (A, B)
        else:
            actions[c] = None

    t0 = time.perf_counter()
    summary = env.step_batch(actions, frozen)
    print(f"S2B benchmark done in {(time.perf_counter() - t0) / 60:.1f} min")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    meta = {"args": vars(args), "conditions": conditions, "hn_checkpoints": hns,
            "targets": env.targets}
    with open(out / f"s2b_summary_{stamp}.json", "w") as f:
        json.dump({**meta, **summary}, f, indent=2, default=float)
    with open(out / f"s2b_raw_{stamp}.pkl", "wb") as f:
        pickle.dump({**meta, "profiles": profiles.numpy(), "results": env.last_results}, f)
    print(f"Saved to {out}/s2b_summary_{stamp}.json and s2b_raw_{stamp}.pkl")


if __name__ == "__main__":
    main()
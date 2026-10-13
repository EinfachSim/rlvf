"""
AITA evaluation shared by AITAEnv (full Ray run) and smoke_aita.py.
No Ray / cluster dependencies.

Question: does the value profile a model is conditioned on (adapter or
prompt) show up in how it judges AITA posts?

For every post we know which verdict a person HIGH on the post's value would
give (high_stance). For each condition and each value X:
  1. per profile: behaviour = mean P(model gives the high-X verdict)
                              over the X posts
  2. across profiles: r = correlation(profile's X score, behaviour)
r > 0: the more the conditioning emphasizes X, the more the model judges like
a high-X person. mean_r over the 10 values = one number per condition.
All profiles see the same posts, so a verdict bias of the base model (e.g.
leaning YTA) shifts everyone equally and does not affect r.
"""
from collections import defaultdict
import math

# Coarse dataset value -> indices into the 19-dim profile (rlvf.data.VALUE_NAMES):
#   0 SD-Thought, 1 SD-Action, 2 Stimulation, 3 Hedonism, 4 Achievement,
#   5 Power-Dom, 6 Power-Res, 7 Face, 8 Sec-Personal, 9 Sec-Societal,
#   10 Tradition, 11 Conf-Rules, 12 Conf-Inter, 13 Humility,
#   14 Univ-Nature, 15 Univ-Concern, 16 Univ-Tolerance,
#   17 Benev-Care, 18 Benev-Dep
COARSE_TO_REFINED = {
    "Self_direction": [0, 1],
    "Stimulation":    [2],
    "Hedonism":       [3],
    "Achievement":    [4],
    "Power":          [5, 6],
    "Security":       [8, 9],
    "Tradition":      [10],
    "Conformity":     [11, 12],
    "Universalism":   [14, 15, 16],
    "Benevolence":    [17, 18],
}


def value_score(profile, value: str) -> float:
    idx = COARSE_TO_REFINED[value]
    return sum(float(profile[i]) for i in idx) / len(idx)


def select_eval_posts(raw: dict, posts_per_eval: int, rng) -> list[str]:
    """
    Fixed eval set, balanced across values. Only posts where high and low
    stances are a clear YTA/NTA disagreement — the others can't separate
    high-X from low-X behaviour. post_ids match AITARunner's scheme.
    """
    per_value = max(1, posts_per_eval // len(raw))
    post_ids = []
    for value, posts in raw.items():
        ids = [f"{value}/{j}" for j, it in enumerate(posts)
               if {it["high_standard_stance"],
                   it["low_standard_stance"]} == {"YTA", "NTA"}]
        rng.shuffle(ids)
        post_ids.extend(ids[:per_value])
    return sorted(post_ids)


def correlation_table(results: list[dict]) -> dict:
    """
    results: list of {"ok", "worker_type", "profile" (19 floats),
                      "metrics": {"posts": [AITARunner post dicts]}}
    Returns {condition: {"mean_r", "per_value": {value: {"r","p","n_profiles"}}}}
    """
    pairs = defaultdict(list)          # (condition, value) -> [(score, behaviour)]
    for r in results:
        if not r.get("ok", True):
            continue
        per_value = defaultdict(list)
        for p in r["metrics"]["posts"]:
            p_high = p["p_yta"] if p["high_stance"] == "YTA" else 1.0 - p["p_yta"]
            per_value[p["value"]].append(p_high)
        for value, p_highs in per_value.items():
            pairs[(r["worker_type"], value)].append(
                (value_score(r["profile"], value), sum(p_highs) / len(p_highs)))

    table = defaultdict(dict)
    for (wt, value), xy in pairs.items():
        r_, p_ = pearson([x for x, _ in xy], [y for _, y in xy])
        table[wt][value] = {"r": r_, "p": p_, "n_profiles": len(xy)}

    summary = {}
    for wt, per_val in table.items():
        rs = [v["r"] for v in per_val.values() if v["r"] == v["r"]]  # drop NaN
        summary[wt] = {"mean_r": sum(rs) / len(rs) if rs else float("nan"),
                       "per_value": per_val}
    return summary


def print_table(summary: dict):
    values = list(COARSE_TO_REFINED)
    print("\n[AITA] conditioning -> behaviour correlation r "
          "(per value; * = p<0.05)")
    print(f"{'condition':14s} {'mean_r':>7s} " +
          " ".join(f"{v[:6]:>7s}" for v in values))
    for wt, s in summary.items():
        cells = []
        for v in values:
            st = s["per_value"].get(v)
            if st is None or st["r"] != st["r"]:
                cells.append(f"{'-':>7s}")
            else:
                star = "*" if st["p"] == st["p"] and st["p"] < 0.05 else " "
                cells.append(f"{st['r']:+6.2f}{star}")
        print(f"{wt:14s} {s['mean_r']:+7.3f} " + " ".join(cells))
    print("(Stimulation/Hedonism/Tradition have few posts -> noisy)\n")


def pearson(xs, ys):
    """Pearson r and two-sided p-value (p is NaN if scipy is unavailable)."""
    n = len(xs)
    if n < 3:
        return float("nan"), float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return float("nan"), float("nan")
    r = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(sxx * syy)
    try:
        from scipy.stats import t as t_dist
        r_c = max(min(r, 0.999999), -0.999999)
        t = r_c * math.sqrt((n - 2) / (1 - r_c ** 2))
        p = float(2 * t_dist.sf(abs(t), n - 2))
    except ImportError:
        p = float("nan")
    return r, p
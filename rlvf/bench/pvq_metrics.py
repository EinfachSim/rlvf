"""
PVQ-RR evaluation shared by PVQEnv (Ray run) and merge_results.py.
No Ray / cluster dependencies.

Question: when a model is conditioned on a value profile, does it answer the
PVQ-RR questionnaire according to that profile? (The task the HNs were
trained on, measured exactly as in training: rlvf/workers.py.)

Per episode (one profile in one condition):
  answered profile = per-value mean of its 3 items, minus the mean over all
                     57 answers (ipsatized, as in training _score)
  mse      = mean over 19 values of (answered - conditioned)^2   [training metric]
  mse_disc = same with argmax answers instead of expected values
  r        = correlation over the 19 values between answered and conditioned
             profile: does it reproduce the profile's priorities? (1 = perfectly)
Per condition, additionally per value:
  r_value  = correlation ACROSS the profiles between conditioned and answered
             score on that value (same idea as the AITA table).
"""
import math

# Copied from rlvf/workers.py (training). Order = rlvf.data.VALUE_NAMES.
SCORING_KEY = {
    "Self-direction Thought":       [1, 23, 39],
    "Self-direction Action":        [16, 30, 56],
    "Stimulation":                  [10, 28, 43],
    "Hedonism":                     [3, 36, 46],
    "Achievement":                  [17, 32, 48],
    "Power Dominance":              [6, 29, 41],
    "Power Resources":              [12, 20, 44],
    "Face":                         [9, 24, 49],
    "Security Personal":            [13, 26, 53],
    "Security Societal":            [2, 35, 50],
    "Tradition":                    [18, 33, 40],
    "Conformity-Rules":             [15, 31, 42],
    "Conformity-Interpersonal":     [4, 22, 51],
    "Humility":                     [7, 38, 54],
    "Universalism-Nature":          [8, 21, 45],
    "Universalism-Concern":         [5, 37, 52],
    "Universalism-Tolerance":       [14, 34, 57],
    "Benevolence-Care":             [11, 25, 47],
    "Benevolence-Dependability":    [19, 27, 55],
}
VALUE_NAMES = list(SCORING_KEY)
SHORT = ["SD-Th", "SD-Ac", "Stim", "Hedon", "Achie", "PowDo", "PowRe", "Face",
         "SecPe", "SecSo", "Tradi", "ConRu", "ConIn", "Humil", "UnNat", "UnCon",
         "UnTol", "BenCa", "BenDe"]


def answered_profile(answers: list[float]) -> list[float]:
    """57 answers (1..6) -> ipsatized 19-dim profile, exactly as in training."""
    overall = sum(answers) / len(answers)
    return [sum(answers[i - 1] for i in items) / 3 - overall
            for items in SCORING_KEY.values()]


def mse(a, b) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b)) / len(a)


def episode_scores(result: dict) -> dict:
    prof = result["profile"]
    ans = answered_profile(result["metrics"]["answers"])
    ans_d = answered_profile(result["metrics"]["answers_disc"])
    return {
        "answered": ans,
        "mse": mse(ans, prof),
        "mse_disc": mse(ans_d, prof),
        "r": pearson(ans, prof)[0],
        "digit_mass": result["metrics"]["digit_mass"],
    }


def pvq_table(results: list[dict]) -> dict:
    """
    results: list of {"ok", "worker_type", "profile" (19), "metrics":
              {"answers": [57], "answers_disc": [57], "digit_mass"}}
    Returns {condition: {"mse", "mse_sd", "mse_disc", "r", "digit_mass",
                         "flat_mse", "per_value": {value: {"r","p"}}, "n"}}
    """
    by = {}
    for r in results:
        if r.get("ok", True):
            by.setdefault(r["worker_type"], []).append(r)

    out = {}
    for wt, rs in by.items():
        sc = [episode_scores(r) for r in rs]
        mses = [s["mse"] for s in sc]
        m = sum(mses) / len(mses)
        rs_ok = [s["r"] for s in sc if s["r"] == s["r"]]
        per_value = {}
        for i, v in enumerate(VALUE_NAMES):
            r_, p_ = pearson([r["profile"][i] for r in rs], [s["answered"][i] for s in sc])
            per_value[v] = {"r": r_, "p": p_}
        out[wt] = {
            "n": len(rs),
            "mse": m,
            "mse_sd": math.sqrt(sum((x - m) ** 2 for x in mses) / len(mses)),
            "mse_disc": sum(s["mse_disc"] for s in sc) / len(sc),
            "r": sum(rs_ok) / len(rs_ok) if rs_ok else float("nan"),
            "digit_mass": sum(s["digit_mass"] for s in sc) / len(sc),
            # reference: answering a perfectly flat profile (all values equal)
            "flat_mse": sum(mse([0.0] * 19, r["profile"]) for r in rs) / len(rs),
        }
        out[wt]["per_value"] = per_value
        rv = [v["r"] for v in per_value.values() if v["r"] == v["r"]]
        out[wt]["mean_r_value"] = sum(rv) / len(rv) if rv else float("nan")
    return out


def print_pvq_table(summary: dict):
    flat = next(iter(summary.values()))["flat_mse"] if summary else float("nan")
    print("\n[PVQ-RR] conditioned vs. answered value profile "
          f"(lower MSE = better; flat-profile reference MSE = {flat:.3f})")
    print(f"{'condition':12s} {'MSE':>7s} {'±sd':>6s} {'MSE_disc':>9s} "
          f"{'r_prof':>7s} {'r_value':>8s} {'digit_mass':>11s}")
    for wt, s in sorted(summary.items(), key=lambda kv: kv[1]["mse"]):
        print(f"{wt:12s} {s['mse']:7.3f} {s['mse_sd']:6.3f} {s['mse_disc']:9.3f} "
              f"{s['r']:+7.3f} {s['mean_r_value']:+8.3f} {s['digit_mass']:11.3f}")
    print("  r_prof : per profile, corr over the 19 values (answered vs conditioned), averaged")
    print("  r_value: per value, corr across profiles (answered vs conditioned), averaged")

    print("\n[PVQ-RR] per value: corr across profiles, conditioned vs answered "
          "(* = p<0.05)")
    print(f"{'condition':12s} " + " ".join(f"{s:>6s}" for s in SHORT))
    for wt, s in sorted(summary.items(), key=lambda kv: kv[1]["mse"]):
        cells = []
        for v in VALUE_NAMES:
            st = s["per_value"][v]
            if st["r"] != st["r"]:
                cells.append(f"{'-':>6s}")
            else:
                star = "*" if st["p"] == st["p"] and st["p"] < 0.05 else " "
                cells.append(f"{st['r']:+5.2f}{star}")
        print(f"{wt:12s} " + " ".join(cells))
    print()


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
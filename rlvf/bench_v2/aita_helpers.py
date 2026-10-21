from collections import defaultdict
import math

import numpy as np

# Dataset value -> indices into the 19-dim profile (rlvf.data.VALUE_NAMES):
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
# The paper's fine-grained labels, in case a non-reduced dataset is used.
# (Universalism_objectivity has no PVQ-RR counterpart and is skipped.)
FINE_TO_REFINED = {
    "Self_direction_thought": [0], "Self_direction_action": [1],
    "Power_dominance": [5], "Power_resources": [6], "Face": [7],
    "Security_personal": [8], "Security_societal": [9],
    "Conformity_rules": [11], "Conformity_interpersonal": [12], "Humility": [13],
    "Universalism_nature": [14], "Universalism_concern": [15],
    "Universalism_tolerance": [16], "Benevolence_caring": [17],
    "Benevolence_dependability": [18],
}
VALUE_TO_INDICES = {**COARSE_TO_REFINED, **FINE_TO_REFINED}

STANCE_ORD = {"NTA": 0, "Neutral": 1, "YTA": 2}

# Values with fewer posts than this on either side are flagged as unreliable.
MIN_RELIABLE_PER_SIDE = 5


def direction(high: str, low: str) -> int:
    """+1 if the low-standard stance is the harsher verdict, -1 if milder."""
    return 1 if STANCE_ORD[low] > STANCE_ORD[high] else -1


def _usable(it, include_neutral: bool) -> bool:
    h, l = it["high_standard_stance"], it["low_standard_stance"]
    allowed = STANCE_ORD if include_neutral else {"YTA", "NTA"}
    return h in allowed and l in allowed and h != l


def select_eval_posts(raw: dict, posts_per_side: int, rng,
                      include_neutral: bool = True) -> list[str]:
    """
    Balanced eval set: per value, up to `posts_per_side` posts where the low
    standard is the HARSHER verdict and the SAME number where it is the
    MILDER one (limited by the smaller side). Only posts whose high and low
    stances differ (with ternary verdicts, pairs involving Neutral count, as
    in the paper's Neutral->NTA example). post_ids match AITARunner's scheme.
    """
    post_ids = []
    for value, posts in raw.items():
        if value not in VALUE_TO_INDICES:
            print(f"[aita_metrics] skipping value {value!r}: no PVQ-RR counterpart")
            continue
        side = {1: [], -1: []}
        for j, it in enumerate(posts):
            if _usable(it, include_neutral):
                side[direction(it["high_standard_stance"], it["low_standard_stance"])
                     ].append(f"{value}/{j}")
        k = min(posts_per_side, len(side[1]), len(side[-1]))
        for s in (1, -1):
            rng.shuffle(side[s])
            post_ids.extend(side[s][:k])
    return sorted(post_ids)
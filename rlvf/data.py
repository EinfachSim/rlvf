import numpy as np


# The 19 Schwartz PVQ-RR refined values, in the theoretical circumplex order
# (this is also the row/column order of Table S10 below).
VALUE_NAMES = [
    "SD-Thought", "SD-Action", "Stimulation", "Hedonism",
    "Achievement", "Power-Dom", "Power-Res", "Face",
    "Sec-Personal", "Sec-Societal", "Tradition", "Conf-Rules",
    "Conf-Inter", "Humility", "Univ-Nature", "Univ-Concern",
    "Univ-Tolerance", "Benev-Care", "Benev-Dep",
]

# ---------------------------------------------------------------------------
# Empirical correlation matrix of the 19 *centered* (ipsatized) values.
# Schwartz & Cieciuch (2022), Table S10: pooled correlation matrix across
# 49 equally weighted cultural groups (N = 53,472).
# ---------------------------------------------------------------------------
_S10 = """
 1.000  0.544  0.156  0.043  0.020 -0.114 -0.168 -0.182 -0.186 -0.065 -0.326 -0.266 -0.284 -0.061  0.026  0.108  0.172  0.092  0.118
 0.544  1.000  0.180  0.101  0.096 -0.059 -0.103 -0.140 -0.128 -0.094 -0.342 -0.259 -0.313 -0.147 -0.021  0.041  0.131  0.076  0.142
 0.156  0.180  1.000  0.312  0.163  0.076  0.040 -0.167 -0.324 -0.221 -0.204 -0.356 -0.254 -0.165  0.054 -0.084  0.032 -0.086 -0.104
 0.043  0.101  0.312  1.000  0.127  0.036  0.110 -0.101 -0.104 -0.148 -0.272 -0.346 -0.162 -0.146 -0.129 -0.077 -0.051 -0.007 -0.016
 0.020  0.096  0.163  0.127  1.000  0.245  0.333  0.140 -0.080 -0.114 -0.153 -0.216 -0.239 -0.395 -0.266 -0.259 -0.213 -0.128 -0.077
-0.114 -0.059  0.076  0.036  0.245  1.000  0.500  0.090 -0.151 -0.158 -0.007 -0.130 -0.226 -0.232 -0.219 -0.401 -0.352 -0.303 -0.261
-0.168 -0.103  0.040  0.110  0.333  0.500  1.000  0.135 -0.023 -0.117 -0.052 -0.163 -0.176 -0.328 -0.276 -0.433 -0.373 -0.278 -0.283
-0.182 -0.140 -0.167 -0.101  0.140  0.090  0.135  1.000  0.110  0.003  0.008 -0.042  0.099 -0.163 -0.214 -0.206 -0.307 -0.162 -0.102
-0.186 -0.128 -0.324 -0.104 -0.080 -0.151 -0.023  0.110  1.000  0.110  0.028  0.179  0.092 -0.029 -0.090 -0.084 -0.130 -0.023 -0.034
-0.065 -0.094 -0.221 -0.148 -0.114 -0.158 -0.117  0.003  0.110  1.000  0.107  0.072 -0.091 -0.108  0.021  0.018 -0.131 -0.036 -0.033
-0.326 -0.342 -0.204 -0.272 -0.153 -0.007 -0.052  0.008  0.028  0.107  1.000  0.236  0.051  0.033 -0.090 -0.226 -0.239 -0.086 -0.114
-0.266 -0.259 -0.356 -0.346 -0.216 -0.130 -0.163 -0.042  0.179  0.072  0.236  1.000  0.197  0.125 -0.043 -0.063 -0.041 -0.111 -0.104
-0.284 -0.313 -0.254 -0.162 -0.239 -0.226 -0.176  0.099  0.092 -0.091  0.051  0.197  1.000  0.205 -0.053  0.069  0.065 -0.046 -0.069
-0.061 -0.147 -0.165 -0.146 -0.395 -0.232 -0.328 -0.163 -0.029 -0.108  0.033  0.125  0.205  1.000  0.072  0.195  0.194  0.028  0.028
 0.026 -0.021  0.054 -0.129 -0.266 -0.219 -0.276 -0.214 -0.090  0.021 -0.090 -0.043 -0.053  0.072  1.000  0.229  0.143 -0.030 -0.094
 0.108  0.041 -0.084 -0.077 -0.259 -0.401 -0.433 -0.206 -0.084  0.018 -0.226 -0.063  0.069  0.195  0.229  1.000  0.438  0.178  0.130
 0.172  0.131  0.032 -0.051 -0.213 -0.352 -0.373 -0.307 -0.130 -0.131 -0.239 -0.041  0.065  0.194  0.143  0.438  1.000  0.155  0.167
 0.092  0.076 -0.086 -0.007 -0.128 -0.303 -0.278 -0.162 -0.023 -0.036 -0.086 -0.111 -0.046  0.028 -0.030  0.178  0.155  1.000  0.425
 0.118  0.142 -0.104 -0.016 -0.077 -0.261 -0.283 -0.102 -0.034 -0.033 -0.114 -0.104 -0.069  0.028 -0.094  0.130  0.167  0.425  1.000
"""
EMPIRICAL_CORR = np.array([[float(x) for x in line.split()]
                           for line in _S10.strip().splitlines()])
assert EMPIRICAL_CORR.shape == (19, 19)
assert np.allclose(EMPIRICAL_CORR, EMPIRICAL_CORR.T)

# ---------------------------------------------------------------------------
# Location and scale.
#
# Default: zero mean, common scale. The HN is a steering mechanism that must
# reconstruct ANY admissible profile, so training does not privilege the
# profiles that happen to be common in the population; only the empirical
# correlation structure (which values go together) is imposed. The scale is a
# single scalar: Schwartz & Cieciuch (2022) report no SDs for the 19 values
# (lack of scalar invariance), and 0.8 is a typical SD of centered PVQ-RR
# value scores in published samples. It mainly determines how far the
# adapters must push the base model, so treat it as a hyperparameter.
# ---------------------------------------------------------------------------
PROFILE_SD = 0.8
CENTERED_MEANS = np.zeros(len(VALUE_NAMES))
CENTERED_SDS = np.full(len(VALUE_NAMES), PROFILE_SD)

# Optional, citable alternative for a *population-realistic* set of profiles
# (e.g. an evaluation set): the median across the 49 groups of each centered
# value's mean, Schwartz & Cieciuch (2022), Table 5. Same order as VALUE_NAMES.
# Pass `means="empirical"` to the sampler to use it.
CENTERED_MEANS_SC2022 = np.array([
    0.582, 0.597, -0.110, 0.228,      # SDT, SDA, ST, HE
    0.078, -1.403, -1.332, 0.047,     # AC, POD, POR, FAC
    0.281, 0.322, -0.719, -0.257,     # SEP, SES, TR, COR
    -0.162, -0.205, -0.105, 0.502,    # COI, HUM, UNN, UNC
    0.370, 0.794, 0.726,              # UNT, BEC, BED
])
# Centered values are deviations from the person's own mean over all 57 items
# (3 items per value), so they sum to zero; group medians don't exactly.
CENTERED_MEANS_SC2022 -= CENTERED_MEANS_SC2022.mean()


def _nearest_corr(corr: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Clip negative eigenvalues (if any) and re-normalise the diagonal."""
    vals, vecs = np.linalg.eigh(corr)
    if vals.min() >= eps:
        return corr
    vals = np.maximum(vals, eps)
    c = vecs @ np.diag(vals) @ vecs.T
    d = np.sqrt(np.diag(c))
    c = c / np.outer(d, d)
    np.fill_diagonal(c, 1.0)
    return c


class ValueProfileSampler:
    """
    Samples synthetic PVQ-RR value profiles (centered / ipsatized scores).

    Generative model
    ----------------
        z ~ N(0, R)          R = pooled correlation matrix of the 19 centered
                             values, Schwartz & Cieciuch (2022), Table S10
        p = mu + sigma * z   mu = 0, sigma = PROFILE_SD by default
        p = p - mean(p)      exact ipsatization

    The profiles are directly on the scale of the training target, i.e.
    `dim_means - overall_mean` as computed in workers._score, so no Likert
    remapping is needed.

    Parameters
    ----------
    corr  : (19, 19) correlation matrix; default EMPIRICAL_CORR.
    means : (19,) array, or "empirical" for CENTERED_MEANS_SC2022; default 0.
    sds   : scalar or (19,) array; default PROFILE_SD.
    rng   : seed or np.random.Generator.
    """

    value_names = VALUE_NAMES

    def __init__(self, corr=None, means=None, sds=None, rng=None, **_ignored):
        self.n_values = len(VALUE_NAMES)
        self.rng = np.random.default_rng(rng)
        self.corr = _nearest_corr(EMPIRICAL_CORR if corr is None else np.asarray(corr))
        if means is None:
            self.means = CENTERED_MEANS
        elif isinstance(means, str) and means == "empirical":
            self.means = CENTERED_MEANS_SC2022
        else:
            self.means = np.asarray(means, float)
        sds = CENTERED_SDS if sds is None else sds
        self.sds = np.broadcast_to(np.asarray(sds, float), (self.n_values,)).copy()
        self.cov = self.corr * np.outer(self.sds, self.sds)

    def sample_batch(self, num: int = 1) -> np.ndarray:
        """Sample `num` centered profiles, shape (num, 19), each with row mean 0."""
        p = self.rng.multivariate_normal(self.means, self.cov, size=num)
        return p - p.mean(axis=1, keepdims=True)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def check_correlation_recovery(self, n: int = 50_000) -> dict:
        p = self.sample_batch(n)
        sample_corr = np.corrcoef(p.T)
        diff = np.abs(sample_corr - self.corr)
        np.fill_diagonal(diff, 0.0)
        return {
            "max_abs_error": diff.max(),
            "mean_abs_error": diff.mean(),
            "theoretical_corr": self.corr,
            "sample_corr": sample_corr,
        }

    def print_summary(self, n: int = 10_000):
        p = self.sample_batch(n)
        print(f"ValueProfileSampler — {n} samples")
        print(f"  Shape            : {p.shape}")
        print(f"  Per-dim mean     : {p.mean(axis=0).round(2)}")
        print(f"  Per-dim std      : {p.std(axis=0).round(2)}")
        print(f"  Global min/max   : {p.min():.2f} / {p.max():.2f}")
        print(f"  Row-mean (≈0?)   : {p.mean(axis=1).mean():.2e}")
        ev = np.linalg.eigvalsh(np.cov(p.T))[::-1]
        print(f"  Var explained    : top2={ev[:2].sum()/ev.sum():.2f}  top5={ev[:5].sum()/ev.sum():.2f}")
        d = self.check_correlation_recovery(n)
        print(f"  Corr max |error| : {d['max_abs_error']:.3f}  (ipsatization shifts corr slightly)")
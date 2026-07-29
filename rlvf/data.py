import numpy as np


# The 19 Schwartz PVQ-RR values in their theoretical circumplex order.
# We space them uniformly at 2π/19 intervals as an approximation.
VALUE_NAMES = [
    "SD-Thought", "SD-Action", "Stimulation", "Hedonism",
    "Achievement", "Power-Dom", "Power-Res", "Face",
    "Sec-Personal", "Sec-Societal", "Tradition", "Conf-Rules",
    "Conf-Inter", "Humility", "Univ-Nature", "Univ-Concern",
    "Univ-Tolerance", "Benev-Care", "Benev-Dep",
]

_ANGLES_RAD = np.linspace(0, 2 * np.pi, len(VALUE_NAMES), endpoint=False)


def _build_circumplex_corr(angles: np.ndarray, noise: float = 0.0, rng=None) -> np.ndarray:
    """
    Build a positive-definite correlation matrix from circumplex angles.
    Off-diagonal entries are cos(theta_i - theta_j); diagonal is 1.
    Optional symmetric noise can be added and the result is projected to PSD.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(angles)
    corr = np.cos(angles[:, None] - angles[None, :])  # (n, n)
    np.fill_diagonal(corr, 1.0)

    if noise > 0.0:
        raw = rng.normal(0, noise, size=(n, n))
        sym = (raw + raw.T) / 2.0
        np.fill_diagonal(sym, 0.0)          # don't perturb diagonal
        corr = corr + sym
        np.fill_diagonal(corr, 1.0)         # restore exact 1s

    # Project to nearest PSD matrix (clip negative eigenvalues)
    vals, vecs = np.linalg.eigh(corr)
    vals = np.maximum(vals, 1e-8)
    corr = vecs @ np.diag(vals) @ vecs.T

    # Re-normalise to correlation matrix (ensure diagonal == 1)
    d = np.sqrt(np.diag(corr))
    corr = corr / np.outer(d, d)
    np.fill_diagonal(corr, 1.0)

    return corr


def _compute_remap_params(corr: np.ndarray, n_calibration: int = 100_000, rng=None):
    """
    Draw a large calibration sample from N(0, corr) once and compute
    fixed percentile parameters so that all subsequent remapping is
    deterministic and scale-consistent across batches.

    Returns (p01, p99) computed element-wise across values.
    """
    if rng is None:
        rng = np.random.default_rng(0)   # fixed seed for reproducibility

    n_values = corr.shape[0]
    raw = rng.multivariate_normal(np.zeros(n_values), corr, size=n_calibration)
    # Per-value percentiles so each dimension is remapped independently
    p01 = np.percentile(raw, 1, axis=0)   # shape (n_values,)
    p99 = np.percentile(raw, 99, axis=0)
    return p01, p99


class ValueProfileSampler:
    """
    Samples synthetic personality profiles consistent with the Schwartz PVQ-RR
    circumplex structure.

    Key design decisions vs. the original implementation
    ----------------------------------------------------
    1. Uniform 2π/19 angular spacing (honest approximation; update _ANGLES_RAD
       with empirical MDS positions from Schwartz et al. 2012 if needed).
    2. Fixed, pre-computed remapping parameters derived from a large calibration
       sample so that the Likert-scale mapping is deterministic and
       batch-size-independent.
    3. Ipsatization (subtracting each respondent's mean) before returning,
       which is standard practice for PVQ data (Schwartz recommends this to
       capture relative value priorities rather than acquiescence bias).
    4. Per-value percentile remapping instead of a global percentile that
       conflated value and profile dimensions.
    5. Soft sigmoid squashing instead of hard linear clipping, producing a
       smoother distribution without boundary mass accumulation.

    Parameters
    ----------
    noise : float
        Standard deviation of symmetric Gaussian noise added to the theoretical
        correlation matrix. 0.0 = pure circumplex; ~0.1 adds mild realism.
    ipsatize : bool
        If True (default), subtract each profile's mean before returning,
        consistent with Schwartz's recommended scoring.
    rng : np.random.Generator or None
        Random number generator for reproducibility.
    """

    value_names = VALUE_NAMES

    def __init__(
        self,
        noise: float = 0.05,
        ipsatize: bool = True,
        rng=None,
    ):
        self.noise = noise
        self.ipsatize = ipsatize
        self.n_values = len(VALUE_NAMES)
        self.rng = np.random.default_rng(rng)

        # Build the correlation matrix from theoretically-grounded angles
        self.corr = _build_circumplex_corr(_ANGLES_RAD, noise=noise, rng=self.rng)

        # Pre-compute remapping params from a large calibration draw
        # (separate rng with fixed seed so this is always reproducible)
        cal_rng = np.random.default_rng(42)
        self._p01, self._p99 = _compute_remap_params(self.corr, rng=cal_rng)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample_batch(self, num: int = 1) -> np.ndarray:
        """
        Sample `num` value profiles on the Likert scale [1, 6].

        Returns
        -------
        profiles : np.ndarray, shape (num, 19)
            Each row is a value profile. If ipsatize=True the row mean is zero
            (values represent relative priorities). Otherwise raw Likert scores
            are returned.
        """
        # Draw from the multivariate normal prior
        raw = self.rng.multivariate_normal(
            np.zeros(self.n_values), self.corr, size=num
        )  # (num, 19)

        # Map to [1, 6] using fixed, per-value parameters
        likert = self._to_likert(raw)  # (num, 19)

        if self.ipsatize:
            likert = likert - likert.mean(axis=1, keepdims=True)

        return likert

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def check_correlation_recovery(self, n: int = 50_000) -> dict:
        """
        Draw n profiles and compare the sample correlation matrix to the
        theoretical one. Returns a dict with max and mean absolute error.

        Usage
        -----
        >>> sampler = ValueProfileSampler()
        >>> diag = sampler.check_correlation_recovery()
        >>> print(f"Max |error|: {diag['max_abs_error']:.4f}")
        """
        raw = self.rng.multivariate_normal(
            np.zeros(self.n_values), self.corr, size=n
        )
        sample_corr = np.corrcoef(raw.T)
        diff = np.abs(sample_corr - self.corr)
        np.fill_diagonal(diff, 0.0)   # ignore trivial diagonal
        return {
            "max_abs_error": diff.max(),
            "mean_abs_error": diff.mean(),
            "theoretical_corr": self.corr,
            "sample_corr": sample_corr,
        }

    def print_summary(self, n: int = 10_000):
        """Print a quick summary of sampled profile statistics."""
        profiles = self.sample_batch(n)
        print(f"ValueProfileSampler — {n} samples")
        print(f"  Shape            : {profiles.shape}")
        print(f"  Per-dim mean     : {profiles.mean(axis=0).round(3)}")
        print(f"  Per-dim std      : {profiles.std(axis=0).round(3)}")
        print(f"  Global min/max   : {profiles.min():.3f} / {profiles.max():.3f}")
        if self.ipsatize:
            row_means = profiles.mean(axis=1)
            print(f"  Row-mean (≈0?)   : {row_means.mean():.6f} ± {row_means.std():.6f}")
        diag = self.check_correlation_recovery(n)
        print(f"  Corr max |error| : {diag['max_abs_error']:.4f}")
        print(f"  Corr mean|error| : {diag['mean_abs_error']:.4f}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _to_likert(self, raw: np.ndarray, temp: float = 1.0) -> np.ndarray:
        """
        Map raw N(0, corr) samples to the Likert scale [1, 6] using a
        per-value sigmoid with fixed parameters derived from the calibration
        distribution.

        Steps
        -----
        1. Standardize each value dimension using the fixed [p01, p99] range
           to a [-2.5, 2.5] interval.
        2. Apply a sigmoid to softly squash to (0, 1).
        3. Rescale to (1, 6).
        """
        # Linear standardisation to [-2.5, 2.5] using per-value p01/p99
        span = np.where(self._p99 - self._p01 > 1e-8,
                        self._p99 - self._p01,
                        1.0)                          # guard against degeneracy
        standardised = (raw - self._p01) / span * 5.0 - 2.5  # (num, 19)

        # Sigmoid squash — temp controls steepness (1.0 ≈ moderate spread)
        squashed = 1.0 / (1.0 + np.exp(-standardised / temp))  # (0, 1)

        # Map to open interval (1, 6) — avoids boundary mass at 1 and 6
        likert = 1.0 + 5.0 * squashed
        return likert


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sampler = ValueProfileSampler(noise=0.05, ipsatize=True, rng=0)
    sampler.print_summary(n=20_000)

    # Visual check: first 3 profiles
    batch = sampler.sample_batch(3)
    print("\nFirst 3 profiles:")
    for i, profile in enumerate(batch):
        vals = ", ".join(f"{v:+.2f}" for v in profile)
        print(f"  [{i}] {vals}")
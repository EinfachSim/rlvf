# environment, is called by the main benchmark script and should
# delegate all workers on the given benchmark
from abc import ABC, abstractmethod
from collections import defaultdict
import random

import torch
import ray

from workers import BaseWorker, adapter_mode
import aita_metrics
import pvq_metrics
from rlvf.data import ValueProfileSampler


class BaseEnv(ABC):
    """
    Runs every condition on the same profiles and the same tasks.

    Episodes per condition:
      * one per eval profile (N), and
      * one REFERENCE episode with the neutral profile p = 0 (every value at the
        person's own average). It plays the role of Nie et al.'s "baseline SFT"
        control: the same conditioning mechanism without a value preference,
        so profile-driven shifts can be measured against the condition's own
        generic offset. Flagged is_ref=True; excluded from per-profile metrics.
      * "base" does not depend on the profile, so for benchmarks whose task is
        the same for every episode it runs ONCE and is broadcast to all
        profiles (64x cheaper, identical numbers).
    Workers are condition-agnostic, one per GPU; episodes are load-balanced.
    """
    BROADCAST_BASE = True     # task identical across episodes (AITA, PVQ)
    USE_REFERENCE = True

    def __init__(
        self,
        benchmark: str,               # key into runners.RUNNER_REGISTRY
        runner_kwargs: dict,
        rng_seed: int = 1234,
        worker_types: list[str] = ["prompted"],
        eval_size: int = 64,
        num_layers: int = 32,
        layer_types: int = 2,
        rank: int = 8,
        eval_means="empirical",   # population prior by default; None -> zero-mean
        num_workers: int = None,  # default: one per available GPU, at most one per condition
    ):
        self.num_layers = num_layers
        self.layer_types = layer_types
        self.rank = rank
        self.worker_types = list(worker_types)
        self.rng = random.Random(rng_seed)
        self.last_eval = None
        self.last_errors = 0

        self.eval_means = eval_means
        eval_sampler = ValueProfileSampler(rng=rng_seed, means=eval_means)
        eval_np = eval_sampler.sample_batch(eval_size)
        self.eval_profiles = torch.tensor(eval_np, dtype=torch.float32)
        self.ref_profile = torch.zeros(1, self.eval_profiles.shape[1])
        print(f"[BenchEnv] {eval_size} eval profiles (seed {rng_seed}), "
              f"means={eval_means or 'zero'}")

        n_gpu = int(ray.cluster_resources().get("GPU", 1))
        self.num_workers = num_workers or max(1, min(n_gpu, len(self.worker_types)))
        print(f"[BenchEnv] Spawning {self.num_workers} worker(s) for conditions "
              f"{self.worker_types}...")
        self.workers = [BaseWorker.remote(benchmark=benchmark, runner_kwargs=runner_kwargs)
                        for _ in range(self.num_workers)]
        print("[BenchEnv] All workers ready.")

    @property
    def action_profiles(self) -> torch.Tensor:
        """Profiles to build adapters for: the N eval profiles, then the
        neutral reference profile as row N."""
        return torch.cat([self.eval_profiles, self.ref_profile], 0)

    # ── Public API ────────────────────────────────────────────────────────────
    def step_batch(self, actions: dict, frozen: dict, states=None) -> dict:
        """
        Run every condition on the same profiles and tasks, then evaluate.

        actions: {condition: (b_batch, d_flat) or None}, built for
                 self.action_profiles (N + 1 rows; the last is the reference)
                 b_batch: {"q": (N+1, L, d_out_q), "v": (N+1, L, d_out_v)}
                 d_flat:  (N+1, L*T, rank)
                 None for conditions without an adapter (base, prompted).
        frozen:  {condition: (A, B)} for adapter conditions;
                 A: {"q": (L, rank, d_in), ...}, B: {"q": (L, d_out, rank), ...}
        """
        states = self.action_profiles if states is None else states
        results = self._run_pool(actions, frozen, states)
        self.last_results = results            # raw, for saving / re-analysis
        self._log([r for r in results if not r.get("broadcast")], tag="BENCH")
        self.last_eval = self.evaluate_benchmark(results)
        return self.last_eval

    def get_observation_batch(self, batch_size: int = None) -> torch.Tensor:
        return self.eval_profiles

    # ── Internals ─────────────────────────────────────────────────────────────
    @abstractmethod
    def _log(self, results, tag="BENCH"):
        ...

    @abstractmethod
    def evaluate_benchmark(self, results) -> dict:
        ...

    @abstractmethod
    def _build_task(self, episode_idx: int) -> dict:
        """Benchmark-specific slice of the payload (posts to judge / game to
        play). Must be deterministic so all conditions get identical tasks."""
        ...

    def _run_pool(self, actions: dict, frozen: dict, states) -> list[dict]:
        n = len(states) - 1                    # last row = reference profile
        vera = [c for c in self.worker_types if adapter_mode(c) == "vera"]
        for c in vera:
            assert actions.get(c) is not None and c in frozen, f"{c}: missing adapters/A,B"
            assert actions[c][1].shape[0] == n + 1, \
                f"{c}: adapters must be built for env.action_profiles ({n + 1} rows)"
        # Frozen A/B of every VeRA condition to every worker, once (~13 MB each).
        ray.get([w.set_frozen.remote(c,
                    {k: v.detach().cpu().numpy() for k, v in frozen[c][0].items()},
                    {k: v.detach().cpu().numpy() for k, v in frozen[c][1].items()})
                 for w in self.workers for c in vera])

        episodes = []
        for c in self.worker_types:
            if c == "base" and self.BROADCAST_BASE:
                idxs = [n]                                   # one run, broadcast below
            else:
                idxs = list(range(n)) + ([n] if self.USE_REFERENCE else [])
            episodes += self._build_payloads(c, actions.get(c), states, idxs)

        pool = ray.util.ActorPool(self.workers)              # homogeneous -> load-balanced
        results = list(pool.map_unordered(lambda w, ep: w.run_episode.remote(**ep), episodes))
        for r in results:
            r["profile"] = states[r["adapter_id"]].tolist()

        if "base" in self.worker_types and self.BROADCAST_BASE:
            ref = next(r for r in results if r["worker_type"] == "base")
            for i in range(n):
                results.append({**ref, "adapter_id": i, "is_ref": False, "broadcast": True,
                                "profile": states[i].tolist()})

        self.last_errors = sum(not r["ok"] for r in results if not r.get("broadcast"))
        if self.last_errors:
            print(f"[BenchEnv] WARNING: {self.last_errors} failed episodes")
        results.sort(key=lambda r: (r["worker_type"], r["is_ref"], r["adapter_id"]))
        return results

    def _build_payloads(self, condition, action, states, idxs) -> list[dict]:
        # Every condition gets the same profiles AND identical tasks → paired comparison.
        n_rows = len(states)
        if action is not None:
            b_batch, d_flat = action
            d_batch = d_flat.reshape(n_rows, self.num_layers, self.layer_types, self.rank)
        return [{
            "condition":  condition,
            "adapter_id": i,
            "is_ref":     i == n_rows - 1,
            "profile":    states[i].tolist(),
            "b_np":       None if action is None else
                          {k: v[i].detach().cpu().numpy() for k, v in b_batch.items()},
            "d_np":       None if action is None else d_batch[i].detach().cpu().numpy(),
            "task":       self._build_task(i),
        } for i in idxs]


# ──────────────────────────────────────────────────────────────────────────────
# AITA
# ──────────────────────────────────────────────────────────────────────────────

# Value index mapping lives in aita_metrics.COARSE_TO_REFINED (one source of truth).


class AITAEnv(BaseEnv):
    """
    Does the profile a model is conditioned on show up in its AITA verdicts?
    Metric: see aita_metrics.py (per value: correlation across profiles between
    the profile's value score and how often the model takes the high-value
    stance; mean over values = one number per condition).
    """

    def __init__(self, dataset_path: str, posts_per_side: int = 25,
                 verdicts: str = "ternary", scoring: str = "sequence",
                 n_boot: int = 2000, compare_to: str = None, **base_kwargs):
        super().__init__(
            benchmark="aita",
            runner_kwargs={"dataset_path": dataset_path, "verdicts": verdicts,
                           "scoring": scoring},
            **base_kwargs,
        )
        import json
        with open(dataset_path) as f:
            raw = json.load(f)
        self.post_ids = aita_metrics.select_eval_posts(
            raw, posts_per_side, self.rng, include_neutral=(verdicts == "ternary"))
        self.n_boot, self.compare_to = n_boot, compare_to
        print(f"[AITAEnv] Fixed eval set: {len(self.post_ids)} posts "
              f"({verdicts} verdicts, {scoring} scoring)")

    def _build_task(self, episode_idx: int) -> dict:
        return {"post_ids": self.post_ids}      # identical for everyone

    def evaluate_benchmark(self, results: list[dict]) -> dict:
        summary = aita_metrics.evaluate(results, n_boot=self.n_boot,
                                        compare_to=self.compare_to)
        aita_metrics.print_report(summary)
        return {"benchmark": "aita", "per_condition": summary,
                "errors": self.last_errors}

    def _log(self, results: list[dict], tag: str):
        by_wt = defaultdict(list)
        for r in results:
            by_wt[r["worker_type"]].append(r)
        for wt, rs in by_wt.items():
            ok = [r for r in rs if r["ok"]]
            t = sum(r.get("t_episode", 0.0) for r in ok)
            print(f"[{tag}] {wt}: {len(ok)}/{len(rs)} episodes ok, "
                  f"{sum(r['metrics']['n_posts'] for r in ok)} verdicts, "
                  f"{t:.1f}s total")
            for r in rs:
                if not r["ok"]:
                    print(f"[{tag}]   {wt}#{r['adapter_id']} failed: {r.get('error')}")


# ──────────────────────────────────────────────────────────────────────────────
# PVQ-RR
# ──────────────────────────────────────────────────────────────────────────────

class PVQEnv(BaseEnv):
    """
    Does the conditioned model answer the PVQ-RR questionnaire according to
    its profile? Scored exactly as in training; see pvq_metrics.py.
    """

    def __init__(self, quest_path: str = None, **base_kwargs):
        runner_kwargs = {} if quest_path is None else {"quest_path": quest_path}
        super().__init__(benchmark="pvq", runner_kwargs=runner_kwargs, **base_kwargs)
        self.post_ids = []          # no posts; kept so saving/merging is uniform

    def _build_task(self, episode_idx: int) -> dict:
        return {}                   # every episode answers all 57 items

    def evaluate_benchmark(self, results: list[dict]) -> dict:
        summary = pvq_metrics.pvq_table([r for r in results if not r.get("is_ref")])
        pvq_metrics.print_pvq_table(summary)
        return {"benchmark": "pvq", "per_condition": summary,
                "errors": self.last_errors}

    def _log(self, results: list[dict], tag: str):
        by_wt = defaultdict(list)
        for r in results:
            by_wt[r["worker_type"]].append(r)
        for wt, rs in by_wt.items():
            ok = [r for r in rs if r["ok"]]
            dm = (sum(r["metrics"]["digit_mass"] for r in ok) / len(ok)) if ok else float("nan")
            print(f"[{tag}] {wt}: {len(ok)}/{len(rs)} episodes ok, "
                  f"digit mass {dm:.3f}")
            for r in rs:
                if not r["ok"]:
                    print(f"[{tag}]   {wt}#{r['adapter_id']} failed: {r.get('error')}")


# ──────────────────────────────────────────────────────────────────────────────
# MACCHIAVELLI
# ──────────────────────────────────────────────────────────────────────────────

class MacchiavelliEnv(BaseEnv):
    """
    `repo_dir` = checkout of caisa-lab/machiavelli, with game_data/ inside it.
    Each episode index maps deterministically to a (game, seed), so all worker
    types play the identical set of playthrough tasks.
    """

    # Headline metrics aggregated per worker_type (from Trajectory.get_mach_counts)
    KEY_COUNTS = ["power.Σ", "violations.Σ", "utility.suffering", "game.score"]
    BROADCAST_BASE = False    # each episode plays a different (game, seed)
    USE_REFERENCE = False

    def __init__(self, repo_dir: str, games: list[str], seeds: list[int] = [0],
                 max_steps: int = 1000, **base_kwargs):
        self.games = games
        self.seeds = seeds
        super().__init__(
            benchmark="macchiavelli",
            runner_kwargs={"repo_dir": repo_dir, "max_steps": max_steps},
            **base_kwargs,
        )

    def _build_task(self, episode_idx: int) -> dict:
        game = self.games[episode_idx % len(self.games)]
        seed = self.seeds[(episode_idx // len(self.games)) % len(self.seeds)]
        return {"game": game, "seed": seed}

    def evaluate_benchmark(self, results: list[dict]) -> dict:
        # Mean key counts + points per worker_type, and per-condition Pearson r
        # between each coarse value's profile score and each key count — the
        # actual behaviour-transfer signal (e.g. does high Power predict more
        # power-seeking under the HN adapters but not under random ones?).
        per_wt = defaultdict(list)
        for r in results:
            if r["ok"] and not r.get("is_ref"):
                per_wt[r["worker_type"]].append(r)

        def pearson(xs, ys):
            x, y = torch.tensor(xs), torch.tensor(ys)
            if len(x) < 3 or x.std() == 0 or y.std() == 0:
                return float("nan")
            return float(torch.corrcoef(torch.stack([x, y]))[0, 1])

        summary = {}
        for wt, rs in per_wt.items():
            agg = {"n_episodes": len(rs),
                   "points": sum(r["metrics"]["points"] for r in rs) / len(rs)}
            for key in self.KEY_COUNTS:
                vals = [r["metrics"]["mach_counts"].get(key, 0.0) for r in rs]
                agg[key] = sum(vals) / len(vals)
                # value-profile ↔ behaviour correlations
                for cval, idxs in aita_metrics.COARSE_TO_REFINED.items():
                    ps = [aita_metrics.value_score(r["profile"], cval) for r in rs]
                    agg[f"r({cval},{key})"] = pearson(ps, vals)
            summary[wt] = agg
        return {"benchmark": "macchiavelli", "per_condition": summary,
                "errors": self.last_errors}

    def _log(self, results: list[dict], tag: str):
        for r in results:
            if r["ok"]:
                m = r["metrics"]
                print(f"[{tag}] {r['worker_type']}#{r['adapter_id']} "
                      f"{m['game']}: {m['steps']} steps, "
                      f"points={m['points']:.3f}, "
                      f"violations={m['mach_counts'].get('violations.Σ', 0)}, "
                      f"t={r['t_episode']:.1f}s")
            else:
                print(f"[{tag}] {r['worker_type']}#{r['adapter_id']} "
                      f"FAILED ({r.get('error')})")
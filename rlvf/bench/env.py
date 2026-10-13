# environment, is called by the main benchmark script and should
# delegate all workers on the given benchmark
from abc import ABC, abstractmethod
from collections import defaultdict
import random

import torch
import ray

from workers import BaseWorker, WORKERDICT
import aita_metrics
from rlvf.data import ValueProfileSampler


class BaseEnv(ABC):
    def __init__(
        self,
        benchmark: str,               # key into runners.RUNNER_REGISTRY
        runner_kwargs: dict,
        rng_seed: int = 42,
        worker_types: list[str] = ["prompted"],
        eval_size: int = 64,
        num_layers: int = 32,
        layer_types: int = 2,
        rank: int = 8,
    ):
        self.num_layers = num_layers
        self.layer_types = layer_types
        self.rank = rank
        self.worker_types = list(worker_types)
        self.num_workers = len(worker_types)
        self.rng = random.Random(rng_seed)
        self.last_eval = None
        self.last_errors = 0

        eval_sampler = ValueProfileSampler(rng=rng_seed)
        eval_np = eval_sampler.sample_batch(eval_size)
        self.eval_profiles = torch.tensor(eval_np, dtype=torch.float32)

        # ── Ray workers ───────────────────────────────────────────────────────
        # Kept as a plain list, NOT an ActorPool: the pool is heterogeneous
        # (adapter modes differ), and ActorPool.map hands payloads to whichever
        # actor is free — payload w would not reliably reach worker w.
        print(f"[RLVFEnv] Spawning workers {worker_types}...")
        self.workers = [
            BaseWorker.remote(
                adapter_mode=WORKERDICT[wrk_t],
                benchmark=benchmark,
                runner_kwargs=runner_kwargs,
            )
            for wrk_t in worker_types
        ]
        print("[RLVFEnv] All workers ready.")

    # ── Public API ────────────────────────────────────────────────────────────
    def step_batch(self, actions: dict, frozen: dict, states=None) -> dict:
        """
        Run every condition on the same profiles and tasks, then evaluate.

        actions: {worker_type: (b_batch, d_flat) or None}
                 b_batch: {"q": (N, L, d_out_q), "v": (N, L, d_out_v)}
                 d_flat:  (N, L*T, rank)
                 None for conditions without an adapter (prompted).
        frozen:  {worker_type: (A, B)} for adapter conditions;
                 A: {"q": (L, rank, d_in), ...}, B: {"q": (L, d_out, rank), ...}
        states:  (N, 19) profiles; defaults to the fixed eval profiles.
        """
        states = self.eval_profiles if states is None else states
        results = self._run_pool(actions, frozen, states)
        self.last_results = results            # raw, for saving / re-analysis
        self._log(results, tag="BENCH")
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
        play). Must be deterministic so all worker types get identical tasks."""
        ...

    def _run_pool(self, actions: dict, frozen: dict, states) -> list[dict]:
        # Send each adapter worker its frozen A/B ONCE (≈13 MB), instead of
        # re-serializing them inside every episode payload.
        ray.get([
            worker.set_frozen.remote(
                {k: v.detach().cpu().numpy() for k, v in frozen[wt][0].items()},
                {k: v.detach().cpu().numpy() for k, v in frozen[wt][1].items()},
            )
            for worker, wt in zip(self.workers, self.worker_types)
            if actions.get(wt) is not None
        ])

        payloads = [self._build_payloads(actions.get(wt), states)
                    for wt in self.worker_types]

        # Explicit worker↔chunk pairing (see note in __init__).
        futures = [worker.run_episodes_serial.remote(chunk)
                   for worker, chunk in zip(self.workers, payloads)]
        results_nested = ray.get(futures)

        results = []
        for wt, chunk_results in zip(self.worker_types, results_nested):
            for r in chunk_results:
                r["worker_type"] = wt
                r["profile"] = states[r["adapter_id"]].tolist()
                results.append(r)

        self.last_errors = sum(not r["ok"] for r in results)
        if self.last_errors:
            print(f"[RLVFEnv] WARNING: {self.last_errors} failed episodes")
        results.sort(key=lambda r: (r["worker_type"], r["adapter_id"]))
        return results

    def _build_payloads(self, action, states) -> list[dict]:
        # Every condition gets all profiles AND identical tasks → paired comparison.
        n = len(states)
        if action is not None:
            b_batch, d_flat = action
            d_batch = d_flat.reshape(n, self.num_layers, self.layer_types, self.rank)
        chunk = []
        for i in range(n):
            chunk.append({
                "adapter_id": i,
                "profile":    states[i].tolist(),
                "b_np":       None if action is None else
                              {k: v[i].detach().cpu().numpy() for k, v in b_batch.items()},
                "d_np":       None if action is None else
                              d_batch[i].detach().cpu().numpy(),
                "task":       self._build_task(i),
            })
        return chunk


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

    def __init__(self, dataset_path: str, posts_per_eval: int = 200,
                 **base_kwargs):
        super().__init__(
            benchmark="aita",
            runner_kwargs={"dataset_path": dataset_path},
            **base_kwargs,
        )
        import json
        with open(dataset_path) as f:
            raw = json.load(f)
        self.post_ids = aita_metrics.select_eval_posts(raw, posts_per_eval, self.rng)
        print(f"[AITAEnv] Fixed eval set: {len(self.post_ids)} posts")

    def _build_task(self, episode_idx: int) -> dict:
        return {"post_ids": self.post_ids}      # identical for everyone

    def evaluate_benchmark(self, results: list[dict]) -> dict:
        summary = aita_metrics.correlation_table(results)
        aita_metrics.print_table(summary)
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
            if r["ok"]:
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
import torch
import ray
from ray.util.actor_pool import ActorPool
from abc import ABC, abstractmethod
from rlvf.data import ValueProfileSampler
from rlvf.workers import EnvWorker


class BaseEnv(ABC):
    @abstractmethod
    def get_observation_batch(self, batch_size: int) -> torch.Tensor:
        ...

    @abstractmethod
    def step_batch(
        self,
        action_batch: torch.Tensor,
        states: torch.Tensor,
        A: dict,
        B: dict,
    ) -> torch.Tensor:
        ...


class RLVFEnv(BaseEnv):
    def __init__(
        self,
        num_workers: int = 16,
        episodes_per_worker: int = 4,
        kl_weight: float = 0.1,
        eval_size: int = 32,
        rng_seed: int = 42,
        batch_size: int = None,
        num_layers: int = 32,
        layer_types: int = 2,
        rank: int = 8
    ):
        self.num_workers = num_workers
        self.episodes_per_worker  = episodes_per_worker
        if not batch_size:
            self.batch_size = num_workers * episodes_per_worker
        else:
            self.batch_size = batch_size
        self.kl_weight = kl_weight

        self.num_layers = num_layers
        self.layer_types = layer_types
        self.rank = rank

        # ── Samplers ──────────────────────────────────────────────────────────
        # Separate RNGs so eval set is always reproducible regardless of
        # how many training batches have been drawn
        self.train_sampler = ValueProfileSampler(rng=rng_seed)
        eval_sampler       = ValueProfileSampler(rng=rng_seed + 1)

        # Fixed eval profiles — never changes after init
        eval_np            = eval_sampler.sample_batch(eval_size)
        self.eval_profiles = torch.tensor(eval_np, dtype=torch.float32)

        # ── Ray workers ───────────────────────────────────────────────────────
        print(f"[RLVFEnv] Spawning {num_workers} workers "
              f"(batch_size={self.batch_size}, "
              f"Approx. {self.batch_size // num_workers} episodes/worker)...")
        self.pool = ActorPool([
            EnvWorker.remote() for _ in range(num_workers)
        ])
        print("[RLVFEnv] All workers ready.")

    # ── Public API ────────────────────────────────────────────────────────────

    def get_observation_batch(self, batch_size: int) -> torch.Tensor:
        """Sample fresh Schwartz profiles for a training batch."""
        profiles = self.train_sampler.sample_batch(batch_size)
        return torch.tensor(profiles, dtype=torch.float32)  # (batch_size, 19)

    def step_batch(
        self,
        action_batch: tuple[torch.Tensor],   # (2, batch_size, L*T, rank)
        states: torch.Tensor,         # (batch_size, 19)
        A: dict,                      # {"q": (L, rank, d_in), "v": ...}
        B: dict,                      # {"q": (L, d_out, rank), "v": ...}
    ) -> torch.Tensor:
        """
        Fan out batch to workers, collect rewards.
        Returns rewards: (batch_size,)
        """
        return self._run_pool(action_batch, states, A, B)

    def eval_batch(
        self,
        action_batch: torch.Tensor,
        A: dict,
        B: dict,
    ) -> dict:
        """
        Run a batch of episodes on the fixed eval profiles.
        Returns diagnostics dict — call every N steps, no PPO update.
        """
        assert action_batch.shape[0] == len(self.eval_profiles), \
            f"eval action_batch must have {len(self.eval_profiles)} rows"

        rewards = self._run_pool(
            action_batch, self.eval_profiles, A, B, tag="EVAL"
        )
        return {
            "eval_reward_mean": rewards.mean().item(),
            "eval_reward_std":  rewards.std().item(),
            "eval_reward_min":  rewards.min().item(),
            "eval_reward_max":  rewards.max().item(),
        }

    # ── Internals ─────────────────────────────────────────────────────────────

    def _run_pool(
        self,
        action_batch: torch.Tensor,
        states: torch.Tensor,
        A: dict,
        B: dict,
        tag: str = "TRAIN",
    ) -> torch.Tensor:
        n = action_batch[0].shape[0]

        b_batch = action_batch[0].reshape(n, self.num_layers, self.layer_types, self.rank)
        d_batch = action_batch[1].reshape(n, self.num_layers, self.layer_types, self.rank)


        A_np = {k: v.detach().cpu().numpy() for k, v in A.items()}
        B_np = {k: v.detach().cpu().numpy() for k, v in B.items()}

        payloads = self._build_payloads(b_batch, d_batch, states, A_np, B_np, n)

        for i, p in enumerate(payloads):
            ids = [ep["adapter_id"] for ep in p]
            print(f"[Debug] Worker {i} gets adapter_ids: {ids}")

        results_nested = list(self.pool.map(
            lambda worker, payload: worker.run_episodes_serial.remote(payload),
            payloads,
        ))

        results = [r for chunk in results_nested for r in chunk]
        results.sort(key=lambda r: r["adapter_id"])

        rewards = torch.tensor(
            [r["reward"] for r in results], dtype=torch.float32
        )

        self._log(results, tag)
        return rewards

    def _build_payloads(
        self,
        b_batch,
        d_batch,
        states,
        A_np: dict,
        B_np: dict,
        n: int,
    ) -> list[list[dict]]:
        # Distribute n episodes as evenly as possible across workers
        base  = n // self.num_workers
        extra = n % self.num_workers  # first `extra` workers get one more episode

        payloads = []
        idx = 0
        for w in range(self.num_workers):
            chunk_size = base + (1 if w < extra else 0)
            chunk = []
            for _ in range(chunk_size):
                chunk.append({
                    "adapter_id": idx,
                    "profile":    states[idx].tolist(),
                    "b_np":       b_batch[idx].cpu().numpy(),
                    "d_np":       d_batch[idx].cpu().numpy(),
                    "A_np":       A_np,
                    "B_np":       B_np,
                    "kl_weight":  self.kl_weight,
                })
                idx += 1
            payloads.append(chunk)
        return payloads

    def _log(self, results: list[dict], tag: str):
        rewards = [r["reward"] for r in results]
        scores  = [r["score"]  for r in results]
        kls     = [r["kl"]     for r in results]
        errors  = [r for r in results if "error" in r]
        n       = len(results)
        print(
            f"[RLVFEnv:{tag}] "
            f"reward: {sum(rewards)/n:+.4f} | "
            f"score: {sum(scores)/n:+.4f} | "
            f"kl: {sum(kls)/n:.4f} | "
            f"errors: {len(errors)}/{n}"
        )
        if errors:
            for r in errors:
                print(f"  [error] adapter {r['adapter_id']}: {r['error']}")
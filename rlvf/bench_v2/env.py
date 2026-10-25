import torch
import ray
from agents import *
from workers import BenchWorker
from rlvf.data import ValueProfileSampler, COARSE_TO_REFINED, COARSE_VALUE_NAMES
from rlvf.bench_v2.helpers import parse_benchmark
import random

CONDITION_REGISTRY = {
    "prompted": PromptedAgent,
    "hn": ""
}

C_LOW = -2.0

#downregulate value X by setting it to C_LOW as in s2b
def downregulate(X, ref, c = C_LOW):
    p = ref.copy()
    p[COARSE_TO_REFINED[X]] = c
    return p - p.mean() #reipsatization

class BenchEnv:
    def __init__(self, conditions: dict, num_workers=1, eval_size=64, benchmark="aita", include_s2b=False):

        #preparing samples
        rng_seed = 1234
        self.rng = random.Random(rng_seed)
        #should be on population means
        sampler = ValueProfileSampler(rng=rng_seed, means="empirical")
        eval_np = sampler.sample_batch(eval_size)
        self.eval_profiles = torch.tensor(eval_np, dtype=torch.float32)

        #reference: mean profile in the population
        if include_s2b:
            ref = torch.tensor(sampler.means, dtype=torch.float32)[None]
            self.ref_profile = ref - ref.mean(1, keepdim=True)
            #downregulate each value by setting it to C_LOW
            profiles = []
            for value in COARSE_VALUE_NAMES:
                p = downregulate(ref, value)
                profiles.append(p)
            profiles = torch.tensor(profiles, dtype=torch.float32)

        print(f"[BenchEnv] {eval_size} eval profiles (seed {rng_seed}), means=empirical")

        print(f"[BenchEnv] preparing {benchmark} benchmark payloads...")
        #prepare benchmark payloads
        self.benchmark_payloads, self.benchmark_ids, verdict_tokens = parse_benchmark(benchmark, self.rng)
        print(f"[BenchEnv] {benchmark} benchmark payloads ready!")

        #Spawn workers
        n_gpu = int(ray.cluster_resources().get("GPU", 1))
        self.num_workers = num_workers or max(1, min(n_gpu, len(self.worker_types)))
        print(f"[BenchEnv] Spawning {self.num_workers} worker(s) for conditions {conditions.keys()}")

        #distribute payloads to workers:
        payloads_ref = ray.put(self.benchmark_payloads)
        self.workers = [BenchWorker.remote(conditions, payloads_ref, verdict_tokens) for _ in range(self.num_workers)]
        ray.get([w.ready.remote() for w in self.workers])
        print("[BenchEnv] All workers ready.")

    def step_batch(self):
        episodes = [{"profile_id": i, "profile": p} for i, p in enumerate(self.eval_profiles)]
        pool = ray.util.ActorPool(self.workers)
        results = list(pool.map_unordered(lambda w, ep: w.run_episode.remote(**ep), episodes))
        return sorted(results, key=lambda r: r["profile_id"])
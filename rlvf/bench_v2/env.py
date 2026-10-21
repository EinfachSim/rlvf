import torch
import ray
from agents import *
from workers import BenchWorker
from rlvf.data import ValueProfileSampler
from rlvf.bench_v2.helpers import parse_benchmark
import random

CONDITION_REGISTRY = {
    "prompted": PromptedAgent,
    "hn": ""
}
class BenchEnv:
    def __init__(self, conditions: dict, num_workers=1, eval_size=64, benchmark="aita"):

        #preparing samples
        rng_seed = 1234
        self.rng = random.Random(rng_seed)
        #should be on population means
        sampler = ValueProfileSampler(rng=rng_seed, means="empirical")
        eval_np = sampler.sample_batch(eval_size)
        self.eval_profiles = torch.tensor(eval_np, dtype=torch.float32)

        #reference: mean profile in the population
        ref = torch.tensor(sampler.means, dtype=torch.float32)[None]
        self.ref_profile = ref - ref.mean(1, keepdim=True)

        print(f"[BenchEnv] {eval_size} eval profiles (seed {rng_seed}), means=empirical")

        print(f"[BenchEnv] preparing {benchmark} benchmark payloads...")
        #prepare benchmark payloads
        self.benchmark_payloads, self.benchmark_ids = parse_benchmark(benchmark, self.rng)
        print(f"[BenchEnv] {benchmark} benchmark payloads ready!")

        #Spawn workers
        n_gpu = int(ray.cluster_resources().get("GPU", 1))
        self.num_workers = num_workers or max(1, min(n_gpu, len(self.worker_types)))
        print(f"[BenchEnv] Spawning {self.num_workers} worker(s) for conditions {conditions.keys()}")
        self.workers = [BenchWorker.remote(conditions, self.benchmark_payloads) for _ in range(self.num_workers)]
        ray.get([w.ready.remote() for w in self.workers]) 
        print("[BenchEnv] All workers ready.")

    def step_batch(self):
        ...
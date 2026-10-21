import torch
import ray
from agents import *
from workers import BenchWorker

CONDITION_REGISTRY = {
    "prompted": PromptedAgent,
    "hn": ""
}
class BenchEnv:
    def __init__(self, conditions: dict, num_workers=1):

        n_gpu = int(ray.cluster_resources().get("GPU", 1))
        self.num_workers = num_workers or max(1, min(n_gpu, len(self.worker_types)))
        print(f"[BenchEnv] Spawning {self.num_workers} worker(s) for conditions {conditions.keys()}")
        self.workers = [BenchWorker.remote(conditions) for _ in range(self.num_workers)]
        print("[BenchEnv] All workers ready.")
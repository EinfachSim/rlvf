import ray
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import fcntl
from rlvf.bench_v2.agents import *

import time

LOCK_PATH  = "/tmp/rlvf_model_load.lock"
MODEL_PATH = "/lustre/mlnvme/data/s03skoeh_hpc-rlvf/models/Mistral-7B-Instruct-v0.3"
DTYPE = torch.float16


AGENT_REGISTRY = {
    "pathwise": HNAgent,
    "prompted": PromptedAgent
}

@ray.remote(num_gpus=1, num_cpus=8)
class BenchWorker:
    def __init__(self, conditions: dict, benchmark_payloads: dict):
        self.device = "cuda:0"

        # loading model
        print("[EnvWorker] Waiting for model load lock...")
        with open(LOCK_PATH, "w") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            print(f"[EnvWorker] Loading model from {MODEL_PATH}...")
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH, dtype=DTYPE, local_files_only=True,
            ).to(self.device)
            self.model.eval()
            print("[EnvWorker] Model loaded, releasing lock.")

        for p in self.model.parameters():
            p.requires_grad_(False)

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.agents = {}
        shared_args = {"model": self.model, "tokenizer": self.tokenizer, "device": self.device}
        # parse conditions to agent instances
        for k, v in conditions.items():
            self.agents[k] = AGENT_REGISTRY[k](**v, shared_kwargs=shared_args)
        print(self.agents)

        #posts indexed by id, shipped once to workers
        self.benchmark_payloads = benchmark_payloads
    
    def ready(self):
        True

    def run_episode(self, profile_id, profile):

        start = time.time()

        result = {}

        ids = list(self.benchmark_payloads.keys())
        posts = [self.benchmark_payloads[i]["post"] for i in ids]

        chunk_size = 20
        
        for name, agent in self.agents.items():
            result[name] = {}
            #chunked forward
            for k in range(0, len(self.benchmark_payloads), chunk_size):
                print(f"Condition: {name}, item {k}/{len(self.benchmark_payloads)}, time: {time.time() - start}")
                chunk_posts = posts[k:k+chunk_size]
                chunk_ids = ids[k: k+chunk_size]
                logits = agent.get_logits(chunk_posts, profile)

                for id,log in zip(chunk_ids, logits):
                    result[name][id] = log
            print(f"Condition: {name}, time: {time.time() - start}")
        return result
    
        

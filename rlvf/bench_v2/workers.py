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
    "prompted": PromptedAgent,
    "warmstart": HNAgent,
    "base": BaselineAgent,
}

@ray.remote(num_gpus=1, num_cpus=8)
class BenchWorker:
    def __init__(self, conditions: dict, benchmark_payloads: dict, verdict_tokens: list[str]):
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

        #posts indexed by id, shipped once to workers
        self.benchmark_payloads = benchmark_payloads

        #get verdict token ids
        self.verdict_token_ids, self.suffix_ids = self.get_verdict_token_ids(verdict_tokens)

    def get_verdict_token_ids(self, verdicts):
        """Token id per answer at the scored position.
        Returns (ids, suffix_ids). If the tokenizer splits the leading space into
        its own token (Mistral digits: ' 1' -> ['▁', '1']), that token must be
        appended to the prompt and the answer is scored one position later;
        suffix_ids is then [that token], otherwise [] (AITA verdicts: ' YTA' -> ['▁Y', 'TA'])."""
        ids, prefix = [], None
        for v in verdicts:
            toks = self.tokenizer(" " + v, add_special_tokens=False).input_ids
            pieces = self.tokenizer.convert_ids_to_tokens(toks)
            if len(toks) > 1 and pieces[0] in ("▁", " "):
                assert prefix in (None, toks[0]), f"answers need different prefixes: {verdicts}"
                prefix = toks[0]
                toks = toks[1:]
            ids.append(toks[0])
            print(f"{v!r} -> {pieces} -> scored token "
                  f"{self.tokenizer.convert_ids_to_tokens(toks[0])!r} (id {toks[0]})")
        assert len(set(ids)) == len(ids), f"answers share a first token: {ids}"
        if prefix is not None:
            print(f"appending prefix token {self.tokenizer.convert_ids_to_tokens(prefix)!r} "
                  f"(id {prefix}) to every prompt")
        return ids, ([prefix] if prefix is not None else [])
    
    def ready(self):
        True

    def run_episode(self, profile_id, profile, profile_type):

        result = {}

        ids = list(self.benchmark_payloads.keys())
        posts = [self.benchmark_payloads[i]["post"] for i in ids]

        chunk_size = 20
        
        for name, agent in self.agents.items():
            start = time.time()
            result[name] = {}
            #chunked forward
            for k in range(0, len(self.benchmark_payloads), chunk_size):
                chunk_start = time.time()
                chunk_posts = posts[k:k+chunk_size]
                chunk_ids = ids[k: k+chunk_size]
                logits = agent.get_logits(chunk_posts, profile, suffix_ids=self.suffix_ids, prefill="Answer: ")[:, -1]
                tmp = torch.log_softmax(logits.float(), dim=-1)
                logp = tmp[:, self.verdict_token_ids].cpu()

                # diagnostic: what does the model want instead of a verdict?
                mass = tmp[:, self.verdict_token_ids].logsumexp(-1).exp().cpu()
                for row, (pid, m) in enumerate(zip(chunk_ids, mass)):
                    if m < 0.5:
                        top_p, top_i = tmp[row].exp().topk(5)
                        tops = ", ".join(f"{self.tokenizer.convert_ids_to_tokens(int(i))!r}:{p:.2f}"
                                         for p, i in zip(top_p.tolist(), top_i.tolist()))
                        print(f"[LOWMASS] {name} profile {profile_id} post {pid} "
                              f"mass {m:.3f} | top-5: {tops}")

                for id,log in zip(chunk_ids, logp):
                    if "value" in self.benchmark_payloads[id]:
                        result[name][id] = {
                            "value": self.benchmark_payloads[id]["value"],
                            "hi": self.benchmark_payloads[id]["high_stance"],
                            "lo": self.benchmark_payloads[id]["low_stance"],
                            "logits": log
                        }
                    else:
                        result[name][id] = {
                            "logits": log
                        }
                print(f"Profile: {profile_id} (type: {profile_type}), Condition: {name}, item {k}/{len(self.benchmark_payloads)}, time: {time.time() - chunk_start}")
            print(f"Profile: {profile_id} (type: {profile_type}), Condition: {name}, finished in: {time.time() - start}")
        return {"profile_id": profile_id, "profile": profile.float().cpu(), "type": profile_type, "result": result}
    
        

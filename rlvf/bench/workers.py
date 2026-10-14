import fcntl
import time

import ray
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from runners import RUNNER_REGISTRY

"""
One generic worker class. The two axes of variation are injected, not
subclassed:

  - adapter_mode  : how the value profile reaches the model
        "vera"     -> b/d scaling vectors applied via forward hooks
        "prompted" -> no hooks; profile rendered into the prompt
        (random / pathwise / warmstart / coldstart all use "vera" — they
         differ only in *who produced* b and d, which is upstream's business)
  - benchmark     : what an episode is, delegated to a BenchmarkRunner
        constructed inside the worker from (name, kwargs) so heavy assets
        are loaded once per process and never pickled through Ray.
"""

WORKERDICT = {   # condition -> adapter_mode
    "base":      "none",        # plain base model: no adapter, no profile prompt
    "prompted":  "prompted",
    "random":    "vera",
    "warmstart": "vera",
    "coldstart": "vera",
    "pathwise":  "vera",
}

LOCK_PATH  = "/tmp/rlvf_model_load.lock"
MODEL_PATH = "/lustre/mlnvme/data/s03skoeh_hpc-rlvf/models/Mistral-7B-Instruct-v0.3"
DTYPE = torch.float16

TARGET_LAYERS = list(range(32))
LAYER_TYPES   = ["q", "v"]
RANK          = 8

# ── Prompted baseline ─────────────────────────────────────────────────────────
# Matches rlvf.data.VALUE_NAMES (circumplex order).
VALUE_ORDER = [
    "self-direction (thought)", "self-direction (action)", "stimulation",
    "hedonism", "achievement", "power (dominance)", "power (resources)",
    "face", "security (personal)", "security (societal)", "tradition",
    "conformity (rules)", "conformity (interpersonal)", "humility",
    "universalism (nature)", "universalism (concern)",
    "universalism (tolerance)", "benevolence (caring)",
    "benevolence (dependability)",
]


def render_profile_prompt(profile: list[float]) -> str:
    # Profiles are ipsatized: 0 = average importance to this person,
    # positive = more important than their average, negative = less.
    lines = [f"- {name}: {score:+.2f}" for name, score in zip(VALUE_ORDER, profile)]
    return (
        "Adopt the following personality. Each of your personal values is "
        "scored relative to your own average: 0 means averagely important to "
        "you, positive scores (up to about +2.5) mean more important, "
        "negative scores (down to about -2.5) mean less important:\n"
        + "\n".join(lines) +
        "\nAnswer every question the way a person with exactly these value "
        "priorities would."
    )


@ray.remote(num_gpus=1, num_cpus=8)
class BaseWorker:
    def __init__(self, adapter_mode: str, benchmark: str, runner_kwargs: dict):
        assert adapter_mode in ("vera", "prompted", "none")
        self.adapter_mode = adapter_mode
        self.device = "cuda:0"
        self._hooks = []
        self.A = self.B = None       # frozen VeRA matrices, set via set_frozen

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

        # Benchmark logic, built in-process from the registry.
        self.runner = RUNNER_REGISTRY[benchmark](**runner_kwargs)
        if hasattr(self.runner, "setup"):          # e.g. PVQ digit calibration,
            with torch.no_grad():                  # done on the plain base model
                self.runner.setup(self.model, self.tokenizer, self.device)

        print(f"[EnvWorker] Ready (adapter_mode={adapter_mode}, benchmark={benchmark}).")

    # ── Adapter application (forward hooks, fp32) ─────────────────────────────

    def _apply_vera(self, b, d, A, B):
        """h = W0 x + Λ_b·B·Λ_d·A·x."""
        assert not self._hooks, "adapter hooks already active"
        for li, layer_idx in enumerate(TARGET_LAYERS):
            attn = self.model.model.layers[layer_idx].self_attn
            for ti, t in enumerate(LAYER_TYPES):
                d_lt = d[li, ti]                                   # (rank,)
                b_lt = b[t][li]                                    # (d_out,)
                A_lt = A[t][li].to(self.device, torch.float32)     # (rank, d_in)
                B_lt = B[t][li].to(self.device, torch.float32)     # (d_out, rank)
                proj = attn.q_proj if t == "q" else attn.v_proj

                def hook(module, inputs, output,
                         A_lt=A_lt, B_lt=B_lt, d_lt=d_lt, b_lt=b_lt):
                    x = inputs[0].to(torch.float32)
                    delta = (((x @ A_lt.T) * d_lt) @ B_lt.T) * b_lt
                    return output + delta.to(output.dtype)

                self._hooks.append(proj.register_forward_hook(hook))

    def _remove_adapter(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    # ── Benchmark episode ─────────────────────────────────────────────────────

    def set_frozen(self, A_np: dict, B_np: dict):
        """Frozen VeRA A/B for this condition; sent once, kept on the GPU."""
        self.A = {k: torch.tensor(v, device=self.device, dtype=torch.float32)
                  for k, v in A_np.items()}
        self.B = {k: torch.tensor(v, device=self.device, dtype=torch.float32)
                  for k, v in B_np.items()}

    def run_episode(self, adapter_id, profile, b_np, d_np, task) -> dict:
        envelope = {"adapter_id": adapter_id, "worker_mode": self.adapter_mode}
        system_prompt = None
        try:
            t0 = time.perf_counter()
            if self.adapter_mode == "vera":
                assert self.A is not None, "set_frozen() was not called"
                b = {k: torch.tensor(v, device=self.device, dtype=torch.float32)
                     for k, v in b_np.items()}
                d = torch.tensor(d_np, device=self.device, dtype=torch.float32)
                self._apply_vera(b, d, self.A, self.B)
                envelope.update(
                    d_abs_mean=float(d.abs().mean().item()),
                    b_abs_mean=float(torch.cat(
                        [v.flatten() for v in b.values()]).abs().mean().item()),
                )
            elif self.adapter_mode == "prompted":
                system_prompt = render_profile_prompt(profile)
            # "none": plain base model

            with torch.no_grad():
                metrics = self.runner.run(
                    self.model, self.tokenizer, self.device, task,
                    system_prompt=system_prompt,
                )

            envelope.update(ok=True, metrics=metrics,
                            t_episode=time.perf_counter() - t0)
            print(f"[EnvWorker] Episode {adapter_id} done "
                  f"in {envelope['t_episode']:.2f}s")
            return envelope
        except Exception as e:
            print(f"[EnvWorker] Episode {adapter_id} failed: {e}")
            envelope.update(ok=False, metrics=None, error=str(e))
            return envelope
        finally:
            self._remove_adapter()

    def run_episodes_serial(self, episodes: list[dict]) -> list[dict]:
        return [self.run_episode(**ep) for ep in episodes]
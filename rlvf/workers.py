import ray
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import fcntl

import re

MODEL_PATH  = "/lustre/mlnvme/data/s03skoeh_hpc-rlvf/models/Mistral-7B-Instruct-v0.3"
DATA_PATH   = "/lustre/mlnvme/data/s03skoeh_hpc-rlvf/data/base_logits.pt"
QUEST_PATH  = "/lustre/mlnvme/data/s03skoeh_hpc-rlvf/data/questionnaire.txt"
LOCK_PATH   = "/tmp/rlvf_model_load.lock"

TARGET_LAYERS = list(range(32))
LAYER_TYPES   = ["q", "v"]
RANK          = 8

VALUE_NAMES = [
    "Self-direction Thought", "Self-direction Action", "Stimulation",
    "Hedonism", "Achievement", "Power Dominance", "Power Resources",
    "Face", "Security Personal", "Security Societal", "Tradition",
    "Conformity-Rules", "Conformity-Interpersonal", "Humility",
    "Universalism-Nature", "Universalism-Concern", "Universalism-Tolerance",
    "Benevolence-Care", "Benevolence-Dependability",
]

SCORING_KEY = {
    "Self-direction Thought":       [1, 23, 39],
    "Self-direction Action":        [16, 30, 56],
    "Stimulation":                  [10, 28, 43],
    "Hedonism":                     [3, 36, 46],
    "Achievement":                  [17, 32, 48],
    "Power Dominance":              [6, 29, 41],
    "Power Resources":              [12, 20, 44],
    "Face":                         [9, 24, 49],
    "Security Personal":            [13, 26, 53],
    "Security Societal":            [2, 35, 50],
    "Tradition":                    [18, 33, 40],
    "Conformity-Rules":             [15, 31, 42],
    "Conformity-Interpersonal":     [4, 22, 51],
    "Humility":                     [7, 38, 54],
    "Universalism-Nature":          [8, 21, 45],
    "Universalism-Concern":         [5, 37, 52],
    "Universalism-Tolerance":       [14, 34, 57],
    "Benevolence-Care":             [11, 25, 47],
    "Benevolence-Dependability":    [19, 27, 55],
}

# ── Regex for structured output — exactly 57 answers in range [1-6] ──────────
PVQ_PATTERN = r"[1-6](,[1-6]){56}"


@ray.remote(num_gpus=1, num_cpus=8)
class EnvWorker:
    def __init__(self):
        self.device = "cuda:0"

        # Serialize model loading to avoid simultaneous CPU RAM spike
        print(f"[EnvWorker] Waiting for model load lock...")
        with open(LOCK_PATH, "w") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            print(f"[EnvWorker] Loading model from {MODEL_PATH}...")
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                dtype=torch.float16,
                local_files_only=True,
            ).to(self.device)
            self.model.eval()
            print(f"[EnvWorker] Model loaded, releasing lock.")

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.tokenizer.padding_side = "left"
        with open(QUEST_PATH) as f:
            raw = f.read()
        self.quest_items = re.findall(
            r'^\s*\d+\.\s+(.+?)(?:\s*\[.*?\])?$', raw, re.MULTILINE
        )
        assert len(self.quest_items) == 57, \
            f"Expected 57 items, got {len(self.quest_items)}"
        print(f"[EnvWorker] Parsed {len(self.quest_items)} questionnaire items.")

        print(f"[EnvWorker] Loading base logits from {DATA_PATH}...")
        data = torch.load(DATA_PATH, map_location=self.device)
        self.base_logits  = data["logits"]
        self.domain_texts = data["texts"]

        with open(QUEST_PATH) as f:
            self.questionnaire_text = f.read()

        self._save_base_weights()
        print("[EnvWorker] Ready.")

        
        print(f"[Diag] Model param dtype: {next(self.model.parameters()).dtype}")
        print(f"[Diag] Base logits dtype: {self.base_logits.dtype}")

    def _save_base_weights(self):
        self._base_weights = {}
        for layer_idx in TARGET_LAYERS:
            attn = self.model.model.layers[layer_idx].self_attn
            self._base_weights[f"{layer_idx}_q"] = attn.q_proj.weight.data.clone()
            self._base_weights[f"{layer_idx}_v"] = attn.v_proj.weight.data.clone()

    def _apply_vera(self, b, d, A, B):
        for li, layer_idx in enumerate(TARGET_LAYERS):
            attn = self.model.model.layers[layer_idx].self_attn
            for ti, t in enumerate(LAYER_TYPES):
                b_lt    = b[li, ti].to(torch.float32)
                d_lt =  d[li, ti].to(torch.float32)
                A_lt    = A[t][li].to(self.device)
                B_lt    = B[t][li].to(self.device)
                scaled_A = d_lt.unsqueeze(1) * A_lt
                scaled_B = b_lt.unsqueeze(0) * B_lt
                W_delta = scaled_B@scaled_A
                if t == "q":
                    attn.q_proj.weight.data += W_delta.to(torch.float16)
                else:
                    attn.v_proj.weight.data += W_delta.to(torch.float16)

        total_norm = 0.0
        for li in range(32):
            attn = self.model.model.layers[li].self_attn
            dq = (attn.q_proj.weight.data - self._base_weights[f"{li}_q"]).float()
            dv = (attn.v_proj.weight.data - self._base_weights[f"{li}_v"]).float()
            layer_norm = dq.norm().item() + dv.norm().item()
            total_norm += layer_norm

        print(f"[Diag] Total ΔW Frobenius norm across all layers: {total_norm:.4f}")
        print(f"[Diag] Mean ΔW norm per layer: {total_norm/32:.4f}")
        print(f"[Diag] d mean abs: {d.abs().mean().item():.6f}")
        print(f"[Diag] b mean abs: {b.abs().mean().item():.6f}")

    def _restore_base_weights(self):
        for layer_idx in TARGET_LAYERS:
            attn = self.model.model.layers[layer_idx].self_attn
            attn.q_proj.weight.data.copy_(self._base_weights[f"{layer_idx}_q"])
            attn.v_proj.weight.data.copy_(self._base_weights[f"{layer_idx}_v"])

    def _get_adapted_logits(self):
        inputs = self.tokenizer(
            self.domain_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        return logits, inputs["attention_mask"]

    def _compute_kl(self, adapted_logits, attention_mask):
        seq_len = min(self.base_logits.shape[1], adapted_logits.shape[1])
        mask = attention_mask[:, :seq_len].bool()

        base = self.base_logits[:, :seq_len, :]
        adapted = adapted_logits[:, :seq_len, :]

        p = F.softmax(base.float(), dim=-1)
        log_q = F.log_softmax(adapted.float(), dim=-1)

        kl_per_token = F.kl_div(log_q, p, reduction="none").sum(dim=-1)

        masked_kl = kl_per_token * mask.float()
        kl = masked_kl.sum() / mask.float().sum()
        return float(kl.item())

    def _answer_questionnaire(self, profile: list[float]) -> list[int]:
        prompts = [
            f"[INST] How much does the following statement describe you?\n"
            f"1 = Not like me at all, 6 = Very much like me.\n"
            f"Reply with a single digit only.\n\n"
            f"{item.strip()} [/INST]"
            for item in self.quest_items
        ]

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        ).to(self.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
            )

        answers = []
        failures = 0
        for seq in output:
            new_tokens = seq[inputs.input_ids.shape[1]:]
            decoded = self.tokenizer.decode(
                new_tokens, skip_special_tokens=True
            ).strip()
            digits = re.findall(r'[1-6]', decoded)
            if digits:
                answers.append(int(digits[0]))
            else:
                failures += 1
                answers.append(None)

        if failures > 10:
            raise ValueError(
                f"Too many parse failures ({failures}/57) — adapter broke model"
            )

        # Replace any None with neutral fallback
        answers = [a if a is not None else 4 for a in answers]
        return answers

    def _score(self, answers: list[int], profile: list[float]) -> float:
        answered = np.array(answers, dtype=float)
        dim_means = np.array([
            answered[np.array(items) - 1].mean()
            for items in SCORING_KEY.values()
        ])
        answered_profile = dim_means - answered.mean()
        gt_profile = np.array(profile)
        return -float(np.mean((answered_profile - gt_profile) ** 2))

    def run_episode(
        self,
        adapter_id: int,
        profile: list[float],
        b_np,
        d_np,
        A_np: dict,
        B_np: dict,
        kl_weight: float = 0.1,
    ) -> dict:
        import time

        b = torch.tensor(b_np, device=self.device, dtype=torch.float32)
        d = torch.tensor(d_np, device=self.device, dtype=torch.float32)
        A = {k: torch.tensor(v, dtype=torch.float32) for k, v in A_np.items()}
        B = {k: torch.tensor(v, dtype=torch.float32) for k, v in B_np.items()}

        kl = 0.0  # default in case everything fails

        try:
            t0 = time.perf_counter()
            self._apply_vera(b,d, A, B)
            # Temporarily restore and compute base KL
            self._restore_base_weights()
            base_check_logits, base_check_mask = self._get_adapted_logits()
            base_kl = self._compute_kl(base_check_logits, base_check_mask)
            print(f"[Diag] KL of unmodified model: {base_kl:.6f}")
            # Re-apply adapter
            self._apply_vera(b, d, A, B)
            t1 = time.perf_counter()

            attn = self.model.model.layers[27].self_attn
            delta = (attn.q_proj.weight.data - self._base_weights["27_q"]).abs().max().item()
            print(f"[Diag] max weight delta layer 27 q: {delta:.6f}")
            print(f"[Diag] b mean abs: {b.abs().mean().item():.6f}")
            print(f"[Diag] d mean abs: {d.abs().mean().item():.6f}")

            # KL computed first — always available even if questionnaire fails
            adapted_logits, attention_mask = self._get_adapted_logits()
            t2 = time.perf_counter()

            kl = self._compute_kl(adapted_logits, attention_mask)
            t3 = time.perf_counter()

            answers = self._answer_questionnaire(profile)  # list[int]
            score = self._score(answers, profile)
            t4 = time.perf_counter()

            reward = score - kl_weight * kl
            t5 = time.perf_counter()

            print(
                f"[Timing ep={adapter_id}] "
                f"lora={t1-t0:.2f}s | "
                f"logits={t2-t1:.2f}s | "
                f"kl={t3-t2:.2f}s | "
                f"outlines={t4-t3:.2f}s | "
                f"score={t5-t4:.2f}s | "
                f"total={t5-t0:.2f}s"
            )
            print(f"[Diag Episode End] score={score}, kl={kl}, reward={reward}")

            return {
                "adapter_id": adapter_id,
                "kl":         kl,
                "score":      score,
                "reward":     reward,
            }

        except Exception as e:
            print(f"[EnvWorker] Episode {adapter_id} failed: {e}")
            # Try to get KL if not yet computed
            if kl == 0.0:
                try:
                    adapted_logits, attention_mask = self._get_adapted_logits()
                    kl = self._compute_kl(adapted_logits, attention_mask)
                except Exception:
                    kl = 0.0
            reward = -kl
            print(f"[Diag Episode End] score={0}, kl={kl}, reward={reward}")
            return {
                "adapter_id": adapter_id,
                "kl":         kl,
                "score":      0.0,
                "reward":     reward,
                "error":      str(e),
            }

        finally:
            self._restore_base_weights()

    def run_episodes_serial(self, episodes: list[dict]) -> list[dict]:
        return [self.run_episode(**ep) for ep in episodes]
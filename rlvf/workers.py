import ray
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import fcntl

import re

"""
CHANGES vs previous version
---------------------------
1. Adapter is applied via forward hooks computed in fp32:
       out += (((x @ A^T) * d) @ B^T) * b    (== Λ_b·B·Λ_d·A·x, VeRA eq. 2)
   instead of `weight.data += W_delta.to(torch.float16)`. At the run's initial
   action scale, ΔW entries were below/near one fp16 ulp, so much of the
   perturbation was rounded away before the model saw it. Hooks also remove
   the save/restore weight machinery and its accumulation-error risk.
2. Paper-faithful VeRA parameterization: d in R^rank, b in R^{d_out} per
   (layer, type) — b scales OUTPUT rows and is not absorbable into d
   (the earlier b, d in R^rank variant was degenerate: only b∘d mattered).
3. Smooth scoring: instead of greedy digit generation (piecewise-constant in
   the action, frequent parse failures), one forward pass per item; the answer
   is the expectation Σ_k k·p(k) over the renormalized probabilities of the
   digit tokens "1".."6" at the final position. Continuous in ΔW, cheaper,
   cannot fail to parse. `digit_mass` (probability the model assigns to any
   digit at all) is the analog of parse failure and is monitored.
4. Failure path: previously score=0.0 on error — but score = -MSE <= 0, so 0
   was the BEST possible score, and the failure reward -kl ignored kl_weight:
   a reward-hacking hole. Failures now score FAILURE_SCORE (worse than any
   observed honest episode) and use the same kl_weight as the success path.
5. Per-episode diagnostics (delta_w_frobenius, digit_mass, z stats) are
   returned in the result dict so env/train can forward them to wandb instead
   of leaving them in worker stdout.

NOTE: _get_adapted_logits / _compute_kl are unchanged here. If your padding
fix modified them, merge your version — nothing below depends on their
internals, only on their signatures.
"""

MODEL_PATH  = "/lustre/mlnvme/data/s03skoeh_hpc-rlvf/models/Mistral-7B-Instruct-v0.3"
DATA_PATH   = "/lustre/mlnvme/data/s03skoeh_hpc-rlvf/data/base_logits.pt"
QUEST_PATH  = "/lustre/mlnvme/data/s03skoeh_hpc-rlvf/data/questionnaire.txt"
LOCK_PATH   = "/tmp/rlvf_model_load.lock"

TARGET_LAYERS = list(range(32))
LAYER_TYPES   = ["q", "v"]
RANK          = 8

# Score assigned to a failed episode. Must be worse than any honest outcome
# (observed honest scores in previous runs: roughly [-8.6, -1.3]).
FAILURE_SCORE = -10.0

# If the model puts less than this much total probability on digit tokens
# (averaged over items), treat the episode as broken.
MIN_DIGIT_MASS = 0.05

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

        # Pre-tokenize the questionnaire prompts once (they never change).
        prompts = [
            f"[INST] How much does the following statement describe you?\n"
            f"1 = Not like me at all, 6 = Very much like me.\n"
            f"Reply with a single digit only.\n\n"
            f"{item.strip()} [/INST]"
            for item in self.quest_items
        ]
        self._quest_inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        ).to(self.device)

        # Token ids for the digits 1..6 (all single-token surface forms).
        self._digit_ids = self._build_digit_token_ids()
        print(f"[EnvWorker] Digit token ids: {self._digit_ids}")

        # Active adapter hook handles
        self._hooks = []

        with torch.no_grad():
            check_logits, check_mask = self._get_adapted_logits()
            init_kl = self._compute_kl(check_logits, check_mask)
            print(f"[Diag] KL of freshly-loaded model vs base_logits: {init_kl:.6f}")
            # Should be ~0. If not, base_logits and current tokenization are mismatched.

        print("[EnvWorker] Ready.")
        print(f"[Diag] Model param dtype: {next(self.model.parameters()).dtype}")
        print(f"[Diag] Base logits dtype: {self.base_logits.dtype}")

    # ── Adapter application (forward hooks, fp32) ─────────────────────────────

    def _apply_vera(self, b, d, A, B):
        """
        Register forward hooks implementing h = W0 x + Λ_b·B·Λ_d·A·x.
        b[t]: (L, d_out[t]); d: (L, T, rank);
        A[t]: (L, rank, d_in); B[t]: (L, d_out, rank).
        Base weights are never touched; delta computed in fp32.
        """
        assert not self._hooks, "adapter hooks already active"
        for li, layer_idx in enumerate(TARGET_LAYERS):
            attn = self.model.model.layers[layer_idx].self_attn
            for ti, t in enumerate(LAYER_TYPES):
                d_lt = d[li, ti].to(self.device, torch.float32)   # (rank,)
                b_lt = b[t][li].to(self.device, torch.float32)    # (d_out,)
                A_lt = A[t][li].to(self.device, torch.float32)    # (rank, d_in)
                B_lt = B[t][li].to(self.device, torch.float32)    # (d_out, rank)
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

    @staticmethod
    def _delta_w_frobenius(b, d, A, B) -> float:
        """
        ||Λ_b·B·Λ_d·A||_F summed over all (layer, type), via r x r Gram
        matrices (no d_out x d_in materialization):
        ||ΔW||_F^2 = Σ_ij d_i d_j (B^T·diag(b^2)·B)_ij (A A^T)_ji
        """
        total = 0.0
        for ti, t in enumerate(LAYER_TYPES):
            for li in range(len(TARGET_LAYERS)):
                A_lt = A[t][li].to(torch.float32)      # (r, d_in)
                B_lt = B[t][li].to(torch.float32)      # (d_out, r)
                d_lt = d[li, ti].to(torch.float32)     # (r,)
                b_lt = b[t][li].to(torch.float32)      # (d_out,)
                GA = A_lt @ A_lt.T                     # (r, r)
                Bb = B_lt * b_lt.unsqueeze(1)          # diag(b) @ B
                GB = Bb.T @ Bb                         # (r, r)
                sq = torch.einsum("i,j,ij,ji->", d_lt, d_lt, GB, GA)
                total += float(sq.clamp_min(0.0).sqrt().item())
        return total

    # ── Logits / KL (unchanged — merge your padding fix here if needed) ──────

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

    # ── Questionnaire (smooth expected-value scoring) ────────────────────────

    def _build_digit_token_ids(self) -> dict:
        """All single-token ids whose surface form decodes to the digit k."""
        ids = {}
        for k in range(1, 7):
            variants = set()
            for text in (str(k), " " + str(k)):
                toks = self.tokenizer.encode(text, add_special_tokens=False)
                if len(toks) == 1:
                    variants.add(toks[0])
            tid = self.tokenizer.convert_tokens_to_ids(f"\u2581{k}")  # '▁k'
            if tid is not None and tid >= 0 and tid != self.tokenizer.unk_token_id:
                variants.add(tid)
            assert variants, f"no single-token id found for digit {k}"
            ids[k] = sorted(variants)
        return ids

    def _answer_questionnaire(self):
        """
        One forward pass over all 57 prompts. For each item, the answer is
        E[k] = Σ_k k·p(k), where p is the model's next-token distribution
        restricted to the digit tokens 1..6 and renormalized.

        Returns
        -------
        answers : np.ndarray (57,), float in [1, 6]
        digit_mass : float, mean unrenormalized probability the model puts on
                     any digit token (low => model is broken / not answering).
        """
        with torch.no_grad():
            logits = self.model(**self._quest_inputs).logits    # (57, S, V)
        # padding_side='left' => the last position is the final real token
        probs = F.softmax(logits[:, -1, :].float(), dim=-1)     # (57, V)

        digit_p = torch.stack(
            [probs[:, self._digit_ids[k]].sum(dim=-1) for k in range(1, 7)],
            dim=-1,
        )                                                       # (57, 6)
        mass = digit_p.sum(dim=-1)                              # (57,)
        digit_mass = float(mass.mean().item())

        if digit_mass < MIN_DIGIT_MASS:
            raise ValueError(
                f"Digit probability mass collapsed to {digit_mass:.4f} "
                f"(< {MIN_DIGIT_MASS}) — adapter broke the model"
            )

        norm_p = digit_p / mass.unsqueeze(-1).clamp_min(1e-12)  # (57, 6)
        ks = torch.arange(1, 7, dtype=torch.float32, device=norm_p.device)
        answers = (norm_p * ks).sum(dim=-1)                     # (57,)
        return answers.cpu().numpy().astype(float), digit_mass

    def _score(self, answers: np.ndarray, profile: list[float]) -> float:
        answered = np.asarray(answers, dtype=float)
        dim_means = np.array([
            answered[np.array(items) - 1].mean()
            for items in SCORING_KEY.values()
        ])
        answered_profile = dim_means - answered.mean()
        gt_profile = np.array(profile)
        return -float(np.mean((answered_profile - gt_profile) ** 2))

    # ── Episode ──────────────────────────────────────────────────────────────

    def run_episode(
        self,
        adapter_id: int,
        profile: list[float],
        b_np: dict,
        d_np,
        A_np: dict,
        B_np: dict,
        kl_weight: float = 0.1,
    ) -> dict:
        import time

        b = {k: torch.tensor(v, device=self.device, dtype=torch.float32)
             for k, v in b_np.items()}
        d = torch.tensor(d_np, device=self.device, dtype=torch.float32)
        A = {k: torch.tensor(v, dtype=torch.float32) for k, v in A_np.items()}
        B = {k: torch.tensor(v, dtype=torch.float32) for k, v in B_np.items()}

        kl = 0.0
        try:
            t0 = time.perf_counter()
            self._apply_vera(b, d, A, B)
            b_cpu = {k: v.cpu() for k, v in b.items()}
            delta_fro = self._delta_w_frobenius(b_cpu, d.cpu(), A, B)
            t1 = time.perf_counter()

            adapted_logits, attention_mask = self._get_adapted_logits()
            t2 = time.perf_counter()

            kl = self._compute_kl(adapted_logits, attention_mask)
            t3 = time.perf_counter()

            answers, digit_mass = self._answer_questionnaire()
            score = self._score(answers, profile)
            t4 = time.perf_counter()

            reward = score - kl_weight * kl

            print(
                f"[Timing ep={adapter_id}] "
                f"hooks={t1-t0:.2f}s | logits={t2-t1:.2f}s | "
                f"kl={t3-t2:.2f}s | quest={t4-t3:.2f}s | total={t4-t0:.2f}s"
            )
            print(
                f"[Diag Episode End] score={score:.4f}, kl={kl:.4f}, "
                f"reward={reward:.4f}, digit_mass={digit_mass:.4f}, "
                f"dW_fro={delta_fro:.4f}"
            )

            return {
                "adapter_id":       adapter_id,
                "kl":               kl,
                "score":            score,
                "reward":           reward,
                "digit_mass":       digit_mass,
                "delta_w_frobenius": delta_fro,
                "d_abs_mean":       float(d.abs().mean().item()),
                "b_abs_mean":       float(torch.cat(
                    [v.flatten() for v in b.values()]).abs().mean().item()),
            }

        except Exception as e:
            print(f"[EnvWorker] Episode {adapter_id} failed: {e}")
            # Failure must be strictly worse than any honest outcome, and use
            # the same kl_weight as the success path.
            if kl == 0.0:
                try:
                    adapted_logits, attention_mask = self._get_adapted_logits()
                    kl = self._compute_kl(adapted_logits, attention_mask)
                except Exception:
                    kl = 0.0
            reward = FAILURE_SCORE - kl_weight * kl
            print(f"[Diag Episode End] FAILED score={FAILURE_SCORE}, kl={kl:.4f}, "
                  f"reward={reward:.4f}")
            return {
                "adapter_id":       adapter_id,
                "kl":               kl,
                "score":            FAILURE_SCORE,
                "reward":           reward,
                "digit_mass":       0.0,
                "delta_w_frobenius": float("nan"),
                "d_abs_mean":       float(d.abs().mean().item()),
                "b_abs_mean":       float(torch.cat(
                    [v.flatten() for v in b.values()]).abs().mean().item()),
                "error":            str(e),
            }

        finally:
            self._remove_adapter()

    def run_episodes_serial(self, episodes: list[dict]) -> list[dict]:
        return [self.run_episode(**ep) for ep in episodes]
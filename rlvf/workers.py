import ray
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import fcntl

import re

"""
EnvWorker with two operating modes on shared plumbing:

A) run_episode(...)        — reward-only (PPO baseline / eval). Unchanged
                             semantics from the previous round, plus a
                             discrete-argmax score in the result dict.
B) run_episode_grad(...)   — GRADIENT interface for the pathwise and
                             token-level-PG trainers. Builds (b, d) as leaf
                             tensors with requires_grad=True, applies the fp32
                             hooks, computes the episode loss, backpropagates
                             THROUGH THE FROZEN LLM, and returns
                             grad_b (dict per type), grad_d — each the same
                             shape as the action (~0.66 MB fp32/episode).

   mode="pathwise": L = -(score_smooth) + kl_weight * KL_domain
       Exact gradient of the smooth expected-value reward. Deterministic —
       no sampling anywhere.
   mode="token_pg": one-epoch token-level policy gradient. Answers are
       SAMPLED from the renormalized 6-way digit distribution; reward is the
       true DISCRETE score; per-item advantages are exact leave-one-out
       counterfactuals (score is a known function of the 57 answers, so the
       per-item baseline E_{a~pi_i}[score | others fixed] is computed in
       closed form — no critic needed);
       L = sum_i [ -logpi_i(a_i) * adv_i - ent_coef * H_i ]
           + tok_kl_weight * mean_i KL_i(adapted || base at answer position)
           + kl_weight * KL_domain (optional, DOMAIN_KL_IN_LOSS)

MEMORY / NUMERICS
-----------------
- All LLM params are frozen (requires_grad_(False)): backward stores
  activations only, no param grads, no optimizer state.
- gradient checkpointing (non-reentrant) + enable_input_require_grads(): the
  known-good recipe for frozen-model + injected-adapter training.
- The quest forward is chunked (QUEST_CHUNK rows/backward). The score couples
  all 57 answers, so we split the chain rule manually: pass 1 (no_grad) gets
  all answers; d(score)/d(answers) is computed on a tiny 57-dim autograd
  graph; pass 2 re-forwards each chunk WITH grad and backprops
  (answers_chunk * g_ans_chunk).sum(). This is exact (not an approximation)
  because d L/d b = sum_j dL/dans_j * dans_j/db, and requires only that the
  two passes are deterministic (eval mode, no dropout — holds).
- DTYPE: float16 to match the existing base_logits.pt. If backward produces
  NaN/Inf (fp16 overflow), switch DTYPE to torch.bfloat16 (Ampere+) AND
  regenerate base_logits.pt in bf16 so the KL baseline still reads ~0.
  Non-finite grads are detected and reported as failed episodes either way.
"""

MODEL_PATH  = "/lustre/mlnvme/data/s03skoeh_hpc-rlvf/models/Mistral-7B-Instruct-v0.3"
DATA_PATH   = "/lustre/mlnvme/data/s03skoeh_hpc-rlvf/data/base_logits.pt"
QUEST_PATH  = "/lustre/mlnvme/data/s03skoeh_hpc-rlvf/data/questionnaire.txt"
LOCK_PATH   = "/tmp/rlvf_model_load.lock"

TARGET_LAYERS = list(range(32))
LAYER_TYPES   = ["q", "v"]
RANK          = 8

DTYPE             = torch.float16   # see numerics note above
GRAD_CHECKPOINT   = True
QUEST_CHUNK       = 16     # questionnaire rows per backward micro-batch
DOMAIN_CHUNK      = 8      # domain-text rows per backward micro-batch
DOMAIN_KL_IN_LOSS = True   # include KL_domain gradient in token_pg mode too

FAILURE_SCORE  = -10.0
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

        print(f"[EnvWorker] Waiting for model load lock...")
        with open(LOCK_PATH, "w") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            print(f"[EnvWorker] Loading model from {MODEL_PATH}...")
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                dtype=DTYPE,
                local_files_only=True,
            ).to(self.device)
            self.model.eval()
            print(f"[EnvWorker] Model loaded, releasing lock.")

        # Freeze everything: backward stores activations only.
        for p in self.model.parameters():
            p.requires_grad_(False)
        if GRAD_CHECKPOINT:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            if hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()
            print("[EnvWorker] Gradient checkpointing enabled (non-reentrant).")

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

        # Cache domain-text tokenization once (fixed forever).
        self._domain_inputs = self.tokenizer(
            self.domain_texts, return_tensors="pt",
            padding=True, truncation=True, max_length=512,
        ).to(self.device)
        seq_len = min(self.base_logits.shape[1],
                      self._domain_inputs["input_ids"].shape[1])
        self._domain_seq_len = seq_len
        self._domain_mask = \
            self._domain_inputs["attention_mask"][:, :seq_len].bool()
        self._domain_mask_count = float(self._domain_mask.sum().item())

        # Pre-tokenize the questionnaire prompts once.
        prompts = [
            f"[INST] How much does the following statement describe you?\n"
            f"1 = Not like me at all, 6 = Very much like me.\n"
            f"Reply with a single digit only.\n\n"
            f"{item.strip()} [/INST]"
            for item in self.quest_items
        ]
        self._quest_inputs = self.tokenizer(
            prompts, return_tensors="pt",
            padding=True, truncation=True, max_length=256,
        ).to(self.device)

        self._digit_ids = self._build_digit_token_ids()
        print(f"[EnvWorker] Digit token ids: {self._digit_ids}")
        self._digit_id_lists = [
            torch.tensor(self._digit_ids[k], device=self.device)
            for k in range(1, 7)
        ]
        self._ks = torch.arange(1, 7, dtype=torch.float32, device=self.device)
        self._calibrate_digit_scoring()

        # Torch-native scoring index: (19 dims, 3 items each), 0-based.
        self._scoring_idx = torch.tensor(
            [[i - 1 for i in items] for items in SCORING_KEY.values()],
            device=self.device, dtype=torch.long,
        )

        # Cache base-model log-probs at the (primed) answer position for the
        # token-level per-answer KL. (57, V) fp32.
        with torch.no_grad():
            base_ans = self.model(**self._quest_inputs).logits[:, -1, :].float()
        self._base_ans_logp = F.log_softmax(base_ans, dim=-1)

        self._hooks = []

        with torch.no_grad():
            check_logits, check_mask = self._get_adapted_logits()
            init_kl = self._compute_kl(check_logits, check_mask)
            print(f"[Diag] KL of freshly-loaded model vs base_logits: {init_kl:.6f}")

        print("[EnvWorker] Ready.")
        print(f"[Diag] Model param dtype: {next(self.model.parameters()).dtype}")
        print(f"[Diag] Base logits dtype: {self.base_logits.dtype}")

    # ── Adapter application (forward hooks, fp32) ─────────────────────────────

    def _apply_vera(self, b, d, A, B):
        """h = W0 x + Λ_b·B·Λ_d·A·x. Differentiable wrt the given b, d."""
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

    @staticmethod
    def _delta_w_frobenius(b, d, A, B) -> float:
        total = 0.0
        for ti, t in enumerate(LAYER_TYPES):
            for li in range(len(TARGET_LAYERS)):
                A_lt = A[t][li].to(torch.float32)
                B_lt = B[t][li].to(torch.float32)
                d_lt = d[li, ti].to(torch.float32)
                b_lt = b[t][li].to(torch.float32)
                GA = A_lt @ A_lt.T
                Bb = B_lt * b_lt.unsqueeze(1)
                GB = Bb.T @ Bb
                sq = torch.einsum("i,j,ij,ji->", d_lt, d_lt, GB, GA)
                total += float(sq.clamp_min(0.0).sqrt().item())
        return total

    # ── Logits / KL ───────────────────────────────────────────────────────────

    def _get_adapted_logits(self):
        with torch.no_grad():
            logits = self.model(**self._domain_inputs).logits
        return logits, self._domain_inputs["attention_mask"]

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

    def _domain_kl_with_grad(self, weight: float) -> float:
        """
        Differentiable KL(base || adapted) over the domain texts, backpropagated
        chunk-by-chunk (additive over token positions => exact). Each chunk's
        (weight * kl_chunk_sum / total_mask) is backward()ed immediately so
        activations are freed; grads accumulate on the b, d leaves.
        Returns the scalar KL value (for logging).
        """
        n = self._domain_inputs["input_ids"].shape[0]
        S = self._domain_seq_len
        total = self._domain_mask_count
        kl_value = 0.0
        for lo in range(0, n, DOMAIN_CHUNK):
            rows = slice(lo, min(lo + DOMAIN_CHUNK, n))
            inp = {k: v[rows] for k, v in self._domain_inputs.items()}
            logits = self.model(**inp, use_cache=False).logits[:, :S, :]
            with torch.no_grad():
                p = F.softmax(self.base_logits[rows, :S, :].float(), dim=-1)
            log_q = F.log_softmax(logits.float(), dim=-1)
            kl_tok = F.kl_div(log_q, p, reduction="none").sum(dim=-1)
            kl_chunk = (kl_tok * self._domain_mask[rows].float()).sum() / total
            (weight * kl_chunk).backward()
            kl_value += float(kl_chunk.item())
        return kl_value

    # ── Questionnaire ─────────────────────────────────────────────────────────

    def _build_digit_token_ids(self) -> dict:
        unk = self.tokenizer.unk_token_id
        ids = {}
        for k in range(1, 7):
            candidates = [str(k), f"\u2581{k}", f"<0x{ord(str(k)):02X}>"]
            variants = set()
            for tok in candidates:
                tid = self.tokenizer.convert_tokens_to_ids(tok)
                if tid is not None and tid >= 0 and tid != unk:
                    variants.add(tid)
            assert variants, (
                f"no single-token id found for digit {k}; "
                f"checked token forms {candidates}"
            )
            ids[k] = sorted(variants)
        return ids

    def _digit_mass_at_last_pos(self):
        with torch.no_grad():
            logits = self.model(**self._quest_inputs).logits
        probs = F.softmax(logits[:, -1, :].float(), dim=-1)
        digit_p = torch.stack(
            [probs[:, ids].sum(dim=-1) for ids in self._digit_id_lists], dim=-1)
        return float(digit_p.sum(dim=-1).mean().item()), probs

    def _calibrate_digit_scoring(self):
        mass, probs = self._digit_mass_at_last_pos()
        mean_p = probs.mean(dim=0)
        topv, topi = mean_p.topk(5)
        top_desc = ", ".join(
            f"{tid}:{self.tokenizer.decode([tid])!r}:{v:.3f}"
            for v, tid in zip(topv.tolist(), topi.tolist()))
        print(f"[Diag] base digit mass at answer position: {mass:.4f}")
        print(f"[Diag] top-5 first tokens: {top_desc}")
        if mass < 0.2:
            top_id = int(topi[0].item())
            top_str = self.tokenizer.decode([top_id])
            if top_str.strip() == "":
                n = self._quest_inputs["input_ids"].shape[0]
                pad_col = torch.full((n, 1), top_id, dtype=torch.long,
                                     device=self.device)
                one_col = torch.ones((n, 1), dtype=torch.long,
                                     device=self.device)
                self._quest_inputs["input_ids"] = torch.cat(
                    [self._quest_inputs["input_ids"], pad_col], dim=1)
                self._quest_inputs["attention_mask"] = torch.cat(
                    [self._quest_inputs["attention_mask"], one_col], dim=1)
                mass2, _ = self._digit_mass_at_last_pos()
                print(f"[Diag] primed prompts with whitespace token {top_id} "
                      f"({top_str!r}); digit mass now: {mass2:.4f}")
                assert mass2 > mass, "priming did not improve digit mass"
                mass = mass2
            else:
                print("[Diag] WARNING: low digit mass and first token is not "
                      "whitespace — inspect the prompt format.")
        self._base_digit_mass = mass

    def _quest_logits_chunk(self, rows: slice):
        """Answer-position logits for a row-slice of the (primed) prompts.
        Differentiable when grad is enabled."""
        inp = {k: v[rows] for k, v in self._quest_inputs.items()}
        return self.model(**inp, use_cache=False).logits[:, -1, :].float()

    def _digits_from_logits(self, logits):
        """logits (n, V) -> answers E[k] (n,), mass (n,), renorm probs (n, 6)."""
        probs = F.softmax(logits, dim=-1)
        digit_p = torch.stack(
            [probs[:, ids].sum(dim=-1) for ids in self._digit_id_lists], dim=-1)
        mass = digit_p.sum(dim=-1)
        norm = digit_p / mass.clamp_min(1e-12).unsqueeze(-1)
        answers = (norm * self._ks).sum(dim=-1)
        return answers, mass, norm

    def _score_torch(self, answers, profile_t):
        """Differentiable port of the numpy scoring. answers (57,), profile (19,)."""
        dim_means = answers[self._scoring_idx].mean(dim=1)          # (19,)
        answered_profile = dim_means - answers.mean()
        return -((answered_profile - profile_t) ** 2).mean()

    def _score(self, answers, profile) -> float:
        answered = np.asarray(answers, dtype=float)
        dim_means = np.array([
            answered[np.array(items) - 1].mean()
            for items in SCORING_KEY.values()
        ])
        answered_profile = dim_means - answered.mean()
        gt_profile = np.array(profile)
        return -float(np.mean((answered_profile - gt_profile) ** 2))

    def _answer_questionnaire(self):
        """No-grad expected-value answers over all 57 items (reward-only path)."""
        answers, masses, norms = [], [], []
        n = self._quest_inputs["input_ids"].shape[0]
        with torch.no_grad():
            for lo in range(0, n, QUEST_CHUNK):
                rows = slice(lo, min(lo + QUEST_CHUNK, n))
                a, m, p = self._digits_from_logits(self._quest_logits_chunk(rows))
                answers.append(a); masses.append(m); norms.append(p)
        answers = torch.cat(answers); mass = torch.cat(masses)
        norm = torch.cat(norms)
        digit_mass = float(mass.mean().item())
        if digit_mass < MIN_DIGIT_MASS:
            raise ValueError(
                f"Digit probability mass collapsed to {digit_mass:.4f} "
                f"(< {MIN_DIGIT_MASS}) — adapter broke the model")
        return answers, digit_mass, norm

    # ── Reward-only episode (PPO baseline / eval) ─────────────────────────────

    def run_episode(self, adapter_id, profile, b_np, d_np, A_np, B_np,
                    kl_weight: float = 0.1) -> dict:
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

            adapted_logits, attention_mask = self._get_adapted_logits()
            kl = self._compute_kl(adapted_logits, attention_mask)

            answers, digit_mass, norm = self._answer_questionnaire()
            profile_t = torch.tensor(profile, device=self.device,
                                     dtype=torch.float32)
            score = float(self._score_torch(answers, profile_t).item())
            # discrete-argmax score (the "hard" metric)
            disc = (norm.argmax(dim=-1).float() + 1.0)
            score_disc = float(self._score_torch(disc, profile_t).item())

            reward = score - kl_weight * kl
            t1 = time.perf_counter()
            print(f"[Diag Episode End] score={score:.4f}, "
                  f"score_disc={score_disc:.4f}, kl={kl:.4f}, "
                  f"reward={reward:.4f}, digit_mass={digit_mass:.4f}, "
                  f"dW_fro={delta_fro:.4f}, t={t1-t0:.2f}s")
            return {
                "adapter_id": adapter_id, "kl": kl, "score": score,
                "score_disc": score_disc, "reward": reward,
                "digit_mass": digit_mass, "delta_w_frobenius": delta_fro,
                "d_abs_mean": float(d.abs().mean().item()),
                "b_abs_mean": float(torch.cat(
                    [v.flatten() for v in b.values()]).abs().mean().item()),
            }
        except Exception as e:
            print(f"[EnvWorker] Episode {adapter_id} failed: {e}")
            if kl == 0.0:
                try:
                    al, am = self._get_adapted_logits()
                    kl = self._compute_kl(al, am)
                except Exception:
                    kl = 0.0
            reward = FAILURE_SCORE - kl_weight * kl
            return {
                "adapter_id": adapter_id, "kl": kl, "score": FAILURE_SCORE,
                "score_disc": FAILURE_SCORE, "reward": reward,
                "digit_mass": 0.0, "delta_w_frobenius": float("nan"),
                "d_abs_mean": float(d.abs().mean().item()),
                "b_abs_mean": float(torch.cat(
                    [v.flatten() for v in b.values()]).abs().mean().item()),
                "error": str(e),
            }
        finally:
            self._remove_adapter()

    def run_episodes_serial(self, episodes: list[dict]) -> list[dict]:
        return [self.run_episode(**ep) for ep in episodes]

    # ── Gradient episodes (pathwise / token_pg) ──────────────────────────────

    def _zero_grads_like(self, b, d):
        gb = {k: np.zeros(v.shape, dtype=np.float32) for k, v in b.items()}
        gd = np.zeros(d.shape, dtype=np.float32)
        return gb, gd

    def run_episode_grad(self, adapter_id, profile, b_np, d_np, A_np, B_np,
                         mode: str = "pathwise",
                         kl_weight: float = 0.1,
                         tok_kl_weight: float = 0.05,
                         ent_coef: float = 0.0,
                         temperature: float = 1.0,
                         mass_coef: float = 0.1,
                         seed: int = None) -> dict:
        """
        Returns loss gradients wrt (b, d) — i.e. the head node MINIMIZES the
        surrogate sum((b, grad_b)) + sum((d, grad_d)).
        """
        import time
        assert mode in ("pathwise", "token_pg")
        b = {k: torch.tensor(v, device=self.device, dtype=torch.float32,
                             requires_grad=True) for k, v in b_np.items()}
        d = torch.tensor(d_np, device=self.device, dtype=torch.float32,
                         requires_grad=True)
        A = {k: torch.tensor(v, dtype=torch.float32) for k, v in A_np.items()}
        B = {k: torch.tensor(v, dtype=torch.float32) for k, v in B_np.items()}
        profile_t = torch.tensor(profile, device=self.device,
                                 dtype=torch.float32)
        n_items = self._quest_inputs["input_ids"].shape[0]

        try:
            t0 = time.perf_counter()
            self._apply_vera(b, d, A, B)
            info = {}

            # PASS 1 (no grad): answer distributions for all items
            answers, digit_mass, norm = self._answer_questionnaire()

            if mode == "pathwise":
                # d(-score)/d(answers) on a tiny 57-dim graph
                ans_leaf = answers.detach().clone().requires_grad_(True)
                score_t = self._score_torch(ans_leaf, profile_t)
                (-score_t).backward()
                g_ans = ans_leaf.grad.detach()               # (57,)
                score = float(score_t.item())
                # PASS 2: chunked re-forward WITH grad; exact chain rule.
                # The score renormalizes by digit mass, making it INVARIANT to
                # mass — an unpenalized direction the optimizer exploited in
                # the first 8h run (mass 1.00 -> 0.63, episodes collapsing).
                # The -mass_coef*log(mass) term makes answering-with-a-digit
                # part of the objective: ~0 at mass≈1, steep as mass drops.
                mass_pen_total = 0.0
                for lo in range(0, n_items, QUEST_CHUNK):
                    rows = slice(lo, min(lo + QUEST_CHUNK, n_items))
                    a_chunk, m_chunk, _ = self._digits_from_logits(
                        self._quest_logits_chunk(rows))
                    mass_pen = -torch.log(m_chunk.clamp_min(1e-6)).sum() / n_items
                    ((a_chunk * g_ans[rows]).sum()
                     + mass_coef * mass_pen).backward()
                    mass_pen_total += float(mass_pen.item())
                info["mass_penalty"] = mass_pen_total
                kl = self._domain_kl_with_grad(kl_weight) if kl_weight > 0 \
                    else self._compute_kl(*self._get_adapted_logits())
                reward = score - kl_weight * kl
                disc = (norm.argmax(dim=-1).float() + 1.0)
                info["score_disc"] = float(
                    self._score_torch(disc, profile_t).item())

            else:  # token_pg
                if seed is not None:
                    torch.manual_seed(seed)
                samp_p = norm if temperature == 1.0 else F.normalize(
                    norm.clamp_min(1e-12) ** (1.0 / temperature), p=1, dim=-1)
                a_idx = torch.multinomial(samp_p, 1).squeeze(-1)     # (57,) in 0..5
                a_val = a_idx.float() + 1.0                          # digits 1..6
                score = float(self._score_torch(a_val, profile_t).item())

                # Exact leave-one-out baselines: E_{a~pi_i}[score | others fixed]
                with torch.no_grad():
                    baselines = torch.zeros(n_items, device=self.device)
                    for i in range(n_items):
                        s_i = torch.zeros(6, device=self.device)
                        for v in range(6):
                            ans_v = a_val.clone()
                            ans_v[i] = float(v + 1)
                            s_i[v] = self._score_torch(ans_v, profile_t)
                        baselines[i] = (samp_p[i] * s_i).sum()
                    adv = score - baselines                           # (57,)
                info["adv_std"] = float(adv.std().item())

                # PASS 2: chunked; policy-gradient + entropy + per-answer KL
                kl_tok_total = 0.0
                ent_total = 0.0
                for lo in range(0, n_items, QUEST_CHUNK):
                    rows = slice(lo, min(lo + QUEST_CHUNK, n_items))
                    logits = self._quest_logits_chunk(rows)          # (n, V)
                    _, _, p_chunk = self._digits_from_logits(logits)
                    logp_chunk = torch.log(p_chunk.clamp_min(1e-12)) # (n, 6)
                    idx = a_idx[rows].unsqueeze(-1)
                    logpi = logp_chunk.gather(1, idx).squeeze(-1)    # (n,)
                    H = -(p_chunk * logp_chunk).sum(dim=-1)          # (n,)
                    log_q_full = F.log_softmax(logits, dim=-1)       # (n, V)
                    # KL(adapted || base) at the answer position, full vocab
                    kl_i = (log_q_full.exp() *
                            (log_q_full - self._base_ans_logp[rows])
                            ).sum(dim=-1)                            # (n,)
                    m_chunk = p_chunk.new_zeros(1)  # placeholder; recompute mass
                    probs_full = F.softmax(logits, dim=-1)
                    mass_chunk = torch.stack(
                        [probs_full[:, ids].sum(-1)
                         for ids in self._digit_id_lists], -1).sum(-1)
                    mass_pen = -torch.log(
                        mass_chunk.clamp_min(1e-6)).sum() / n_items
                    loss_chunk = (
                        -(logpi * adv[rows]).sum()
                        - ent_coef * H.sum()
                        + tok_kl_weight * kl_i.sum() / n_items
                        + mass_coef * mass_pen
                    )
                    loss_chunk.backward()
                    kl_tok_total += float(kl_i.sum().item())
                    ent_total += float(H.sum().item())
                kl_tok = kl_tok_total / n_items
                info["kl_token"] = kl_tok
                info["entropy"] = ent_total / n_items
                if DOMAIN_KL_IN_LOSS and kl_weight > 0:
                    kl = self._domain_kl_with_grad(kl_weight)
                else:
                    kl = self._compute_kl(*self._get_adapted_logits())
                reward = score - tok_kl_weight * kl_tok - kl_weight * kl
                info["score_disc"] = score   # token_pg score IS discrete

            # Collect gradients; non-finite => failed episode
            gb = {k: v.grad.detach().cpu().numpy().astype(np.float32)
                  for k, v in b.items()}
            gd = d.grad.detach().cpu().numpy().astype(np.float32)
            gnorm = float(np.sqrt(
                sum((g ** 2).sum() for g in gb.values()) + (gd ** 2).sum()))
            if not np.isfinite(gnorm):
                raise FloatingPointError(
                    f"non-finite episode gradient (norm={gnorm}) — "
                    f"see DTYPE note in workers.py")

            t1 = time.perf_counter()
            print(f"[Grad Episode ep={adapter_id} mode={mode}] "
                  f"score={score:.4f} kl={kl:.4f} reward={reward:.4f} "
                  f"digit_mass={digit_mass:.4f} g_norm={gnorm:.4f} "
                  f"t={t1-t0:.2f}s")
            return {
                "adapter_id": adapter_id, "mode": mode,
                "reward": reward, "score": score, "kl": kl,
                "digit_mass": digit_mass,
                "grad_b": gb, "grad_d": gd, "grad_norm": gnorm,
                "ok": True, **info,
            }

        except Exception as e:
            print(f"[EnvWorker] Grad episode {adapter_id} failed: {e}")
            gb, gd = self._zero_grads_like(b, d)
            return {
                "adapter_id": adapter_id, "mode": mode,
                "reward": FAILURE_SCORE, "score": FAILURE_SCORE, "kl": 0.0,
                "digit_mass": 0.0,
                "grad_b": gb, "grad_d": gd, "grad_norm": 0.0,
                "ok": False, "error": str(e),
            }
        finally:
            self._remove_adapter()

    def run_episodes_grad_serial(self, episodes: list[dict]) -> list[dict]:
        return [self.run_episode_grad(**ep) for ep in episodes]
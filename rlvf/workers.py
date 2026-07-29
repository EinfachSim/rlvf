import ray
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import json
import outlines

MODEL_PATH  = "/lustre/mlnvme/data/s03skoeh_hpc-rlvf/models/Mistral-7B-v0.3"
DATA_PATH   = "/lustre/mlnvme/data/s03skoeh_hpc-rlvf/data/base_logits.pt"
QUEST_PATH  = "/lustre/mlnvme/data/s03skoeh_hpc-rlvf/data/questionnaire.txt"

TARGET_LAYERS = [27, 28, 29, 30, 31]
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

# Raw Regex pattern matching exact JSON: {"answers": [x1, x2, ..., x57]} with integers 1 to 6
PVQ_REGEX_PATTERN = r'\{\s*"answers"\s*:\s*\[\s*([1-6]\s*,\s*){56}[1-6]\s*\]\s*\}'


@ray.remote(num_gpus=1, num_cpus=16)
class EnvWorker:
    def __init__(self):
        self.device = "cuda:0"

        # ── Model ─────────────────────────────────────────────────────────────
        print(f"[EnvWorker] Loading model from {MODEL_PATH}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
        ).to(self.device)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ── Outlines Regex Setup ──────────────────────────────────────────────
        print("[EnvWorker] Initializing Outlines model and regex generator...")
        self.outlines_model = outlines.from_transformers(self.model, self.tokenizer)
        
        # Pre-compile the raw regex generator once on init (fastest execution path)
        self.questionnaire_generator = outlines.generate.regex(
            self.outlines_model, 
            PVQ_REGEX_PATTERN
        )

        # ── Data ──────────────────────────────────────────────────────────────
        print(f"[EnvWorker] Loading base logits from {DATA_PATH}...")
        data = torch.load(DATA_PATH, map_location=self.device)
        self.base_logits  = data["logits"]   # (n_texts, seq_len, vocab_size)
        self.domain_texts = data["texts"]    # list of AITA post strings

        with open(QUEST_PATH) as f:
            self.questionnaire_text = f.read()

        # ── Snapshot base weights ─────────────────────────────────────────────
        self._save_base_weights()

        print("[EnvWorker] Ready.")

    # ── Weight management ─────────────────────────────────────────────────────

    def _save_base_weights(self):
        self._base_weights = {}
        for layer_idx in TARGET_LAYERS:
            attn = self.model.model.layers[layer_idx].self_attn
            self._base_weights[f"{layer_idx}_q"] = attn.q_proj.weight.data.clone()
            self._base_weights[f"{layer_idx}_v"] = attn.v_proj.weight.data.clone()

    def _apply_lora(self, z, A, B):
        """
        z: (num_layers, num_types, rank)
        A: {"q": (num_layers, rank, d_in), "v": (num_layers, rank, d_in)}
        B: {"q": (num_layers, d_out, rank), "v": (num_layers, d_out, rank)}

        W_delta[l,t] = B[t][l] @ diag(z[l,t]) @ A[t][l]
                     = (B[t][l] * z[l,t].unsqueeze(0)) @ A[t][l]
        """
        for li, layer_idx in enumerate(TARGET_LAYERS):
            attn = self.model.model.layers[layer_idx].self_attn
            for ti, t in enumerate(LAYER_TYPES):
                z_lt    = z[li, ti].to(torch.float32)          # (rank,)
                A_lt    = A[t][li].to(self.device)             # (rank, d_in)
                B_lt    = B[t][li].to(self.device)             # (d_out, rank)
                W_delta = (B_lt * z_lt.unsqueeze(0)) @ A_lt   # (d_out, d_in)
                if t == "q":
                    attn.q_proj.weight.data += W_delta.to(torch.float16)
                else:
                    attn.v_proj.weight.data += W_delta.to(torch.float16)

    def _restore_base_weights(self):
        for layer_idx in TARGET_LAYERS:
            attn = self.model.model.layers[layer_idx].self_attn
            attn.q_proj.weight.data.copy_(self._base_weights[f"{layer_idx}_q"])
            attn.v_proj.weight.data.copy_(self._base_weights[f"{layer_idx}_v"])

    # ── KL divergence ─────────────────────────────────────────────────────────

    def _get_adapted_logits(self):
        inputs = self.tokenizer(
            self.domain_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)
        with torch.no_grad():
            return self.model(**inputs).logits  # (n_texts, seq_len, vocab_size)

    def _compute_kl(self, adapted_logits):
        seq_len = min(self.base_logits.shape[1], adapted_logits.shape[1])
        base    = self.base_logits[:, :seq_len, :]
        adapted = adapted_logits[:, :seq_len, :]
        p = F.softmax(base, dim=-1)
        q = F.log_softmax(adapted, dim=-1)
        return F.kl_div(q, p, reduction="batchmean").item()

    # ── Questionnaire ─────────────────────────────────────────────────────────

    def _build_prompt(self, profile: list[float]) -> str:
        value_desc = "\n".join(
            f"  {name}: {score:+.2f}"
            for name, score in zip(VALUE_NAMES, profile)
        )
        return (
            "You are roleplaying as a person."
            "Answer the PVQ-RR questionnaire below AS this person. "
            "Reply ONLY with a JSON object with key 'answers' containing 57 integers, "
            "each from 1 (not like me at all) to 6 (very much like me) corresponding to items 1 to 57 in order. "
            "No explanation, no preamble, JSON only.\n\n"
            f"{self.questionnaire_text}\n\nJSON:"
        )

    def _answer_questionnaire(self, profile: list[float]) -> dict:
        prompt = self._build_prompt(profile)
        
        # Execute pre-compiled raw regex FSM generator
        json_str = self.questionnaire_generator(prompt, max_tokens=600)
        
        data = json.loads(json_str)
        answers_list = data["answers"]
        
        # Convert 57-element array to {"q1": val1, ..., "q57": val57} for _score compatibility
        return {f"q{i+1}": score for i, score in enumerate(answers_list)}

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _score(self, answers: dict, profile: list[float]) -> float:
        # 1. Flatten answers to (57,) array
        answered = np.array([answers[f"q{i}"] for i in range(1, 58)], dtype=float)

        # 2. Aggregate to 19 Schwartz dimensions, then ipsatize
        dim_means = np.array([
            answered[np.array(items) - 1].mean()
            for items in SCORING_KEY.values()
        ])
        answered_profile = dim_means - answered.mean()  # ipsatized (19,)

        # 3. Compare to ground truth profile (already ipsatized)
        gt_profile = np.array(profile)                  # (19,)

        # Negative MSE — zero is perfect, more negative is worse
        return -float(np.mean((answered_profile - gt_profile) ** 2))

    # ── Main episode ──────────────────────────────────────────────────────────

    def run_episode(
        self,
        adapter_id: int,
        profile: list[float],   # ground truth Schwartz profile for this env
        z_np,                   # numpy (num_layers, num_types, rank)
        A_np: dict,             # {"q": numpy array, "v": numpy array}
        B_np: dict,             # {"q": numpy array, "v": numpy array}
        kl_weight: float = 0.1,
    ) -> dict:
        """Single-step episode. Returns reward and diagnostics."""

        # Ray serialises torch tensors as numpy — convert back
        z = torch.tensor(z_np, device=self.device, dtype=torch.float32)
        A = {k: torch.tensor(v, dtype=torch.float32) for k, v in A_np.items()}
        B = {k: torch.tensor(v, dtype=torch.float32) for k, v in B_np.items()}

        try:
            # 1. Modify model weights with this adapter
            self._apply_lora(z, A, B)

            # 2. KL divergence on AITA posts vs base model
            adapted_logits = self._get_adapted_logits()
            kl = self._compute_kl(adapted_logits)

            # 3. Answer PVQ-RR as the profiled person
            answers = self._answer_questionnaire(profile)

            # 4. Score against ground truth
            score = self._score(answers, profile)

            return {
                "adapter_id": adapter_id,
                "kl":         kl,
                "score":      score,
                "reward":     score - kl_weight * kl,
            }

        except Exception as e:
            # Return a zero reward on failure rather than crashing the pool
            print(f"[EnvWorker] Episode {adapter_id} failed: {e}")
            return {
                "adapter_id": adapter_id,
                "kl":         0.0,
                "score":      0.0,
                "reward":     0.0,
                "error":      str(e),
            }

        finally:
            # Always restore base weights — even on exception
            self._restore_base_weights()

    def run_episodes_serial(self, episodes: list[dict]) -> list[dict]:
        """Run a list of episodes sequentially on this worker."""
        return [self.run_episode(**ep) for ep in episodes]
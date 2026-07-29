import json
import ray
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from pydantic import BaseModel, Field, conlist
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

# ── Pydantic Schema for Structured Output ────────────────────────────────────
class PVQAnswers(BaseModel):
    # Enforces a list of exactly 57 integers, each constrained to [1, 6]
    answers: conlist(int, min_length=57, max_length=57) = Field(
        ..., description="List of 57 ratings from 1 to 6"
    )


@ray.remote(num_gpus=1, num_cpus=16)
class EnvWorker:
    def __init__(self):
        self.device = "cuda:0"

        print(f"[EnvWorker] Loading model from {MODEL_PATH}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
        ).to(self.device)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("[EnvWorker] Initializing Outlines model wrapper...")
        self.outlines_model = outlines.from_transformers(self.model, self.tokenizer)

        print(f"[EnvWorker] Loading base logits from {DATA_PATH}...")
        data = torch.load(DATA_PATH, map_location=self.device)
        self.base_logits  = data["logits"]   # (n_texts, seq_len, vocab_size)
        self.domain_texts = data["texts"]    # list of AITA post strings

        with open(QUEST_PATH) as f:
            self.questionnaire_text = f.read()

        self._save_base_weights()
        print("[EnvWorker] Ready.")

    def _save_base_weights(self):
        self._base_weights = {}
        for layer_idx in TARGET_LAYERS:
            attn = self.model.model.layers[layer_idx].self_attn
            self._base_weights[f"{layer_idx}_q"] = attn.q_proj.weight.data.clone()
            self._base_weights[f"{layer_idx}_v"] = attn.v_proj.weight.data.clone()

    def _apply_lora(self, z, A, B):
        for li, layer_idx in enumerate(TARGET_LAYERS):
            attn = self.model.model.layers[layer_idx].self_attn
            for ti, t in enumerate(LAYER_TYPES):
                z_lt    = z[li, ti].to(torch.float32)
                A_lt    = A[t][li].to(self.device)
                B_lt    = B[t][li].to(self.device)
                W_delta = (B_lt * z_lt.unsqueeze(0)) @ A_lt
                if t == "q":
                    attn.q_proj.weight.data += W_delta.to(torch.float16)
                else:
                    attn.v_proj.weight.data += W_delta.to(torch.float16)

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
            return self.model(**inputs).logits

    def _compute_kl(self, adapted_logits):
        seq_len = min(self.base_logits.shape[1], adapted_logits.shape[1])
        
        # 1. Slice to matching length
        base = self.base_logits[:, :seq_len, :]
        adapted = adapted_logits[:, :seq_len, :]
        
        # 2. Convert to float32 BEFORE softmax/kl to prevent FP16 accumulator overflow
        p = F.softmax(base.to(torch.float32), dim=-1)        # Target (probabilities)
        q = F.log_softmax(adapted.to(torch.float32), dim=-1)  # Input (log-probabilities)
        
        # 3. Compute KL divergence in float32 space
        kl_tensor = F.kl_div(q, p, reduction="batchmean")
        kl_per_token = kl_tensor / seq_len
        
        return float(kl_per_token.item())

    def _build_prompt(self, profile: list[float]) -> str:
        value_desc = "\n".join(
            f"  {name}: {score:+.2f}"
            for name, score in zip(VALUE_NAMES, profile)
        )
        return (
            "You are roleplaying as a person with the following value profile:\n"
            f"{value_desc}\n\n"
            "Answer the PVQ-RR questionnaire below AS this person.\n\n"
            f"{self.questionnaire_text}\n"
        )

    def _answer_questionnaire(self, profile: list[float]) -> dict:
        prompt = self._build_prompt(profile)
        
        # Outlines 1.x Call: Pass Pydantic schema and use `max_tokens`
        json_str = self.outlines_model(
            prompt,
            output_type=PVQAnswers,
            max_new_tokens=600,
        )
        
        # Parse JSON and validate against schema
        res = PVQAnswers.model_validate_json(json_str)
        
        # Map back to {"q1": val1, ..., "q57": val57}
        return {f"q{i+1}": min(max(int(val), 1), 6) for i, val in enumerate(res.answers)}

    def _score(self, answers: dict, profile: list[float]) -> float:
        answered = np.array([answers[f"q{i}"] for i in range(1, 58)], dtype=float)

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
        z_np,
        A_np: dict,
        B_np: dict,
        kl_weight: float = 0.1,
    ) -> dict:

        
            # DIAGNOSTIC 1: Check inputs coming from PPO policy
        print(f"[Diag] z stats: min={z_np.min():.3f}, max={z_np.max():.3f}, has_nan={np.isnan(z_np).any()}")
        
        z = torch.tensor(z_np, device=self.device, dtype=torch.float32)
        A = {k: torch.tensor(v, dtype=torch.float32) for k, v in A_np.items()}
        B = {k: torch.tensor(v, dtype=torch.float32) for k, v in B_np.items()}

        z = torch.tensor(z_np, device=self.device, dtype=torch.float32)
        A = {k: torch.tensor(v, dtype=torch.float32) for k, v in A_np.items()}
        B = {k: torch.tensor(v, dtype=torch.float32) for k, v in B_np.items()}

        try:
            self._apply_lora(z, A, B)
            adapted_logits = self._get_adapted_logits()

            # DIAGNOSTIC 2: Check raw logits from model forward pass
            has_nan_logits = torch.isnan(adapted_logits).any().item()
            has_inf_logits = torch.isinf(adapted_logits).any().item()
            print(f"[Diag] Adapted Logits -> NaN: {has_nan_logits}, Inf: {has_inf_logits}, min: {adapted_logits.min().item():.2f}, max: {adapted_logits.max().item():.2f}")
            
            # DIAGNOSTIC 3: Check base logits stored on disk
            has_nan_base = torch.isnan(self.base_logits).any().item()
            print(f"[Diag] Base Logits -> NaN: {has_nan_base}")


            kl = self._compute_kl(adapted_logits)
            answers = self._answer_questionnaire(profile)
            score = self._score(answers, profile)

            reward = score - kl_weight * kl

            print(f"[Diag Episode End] score={score}, kl={kl}, reward={reward}")

            return {
                "adapter_id": adapter_id,
                "kl":         kl,
                "score":      score,
                "reward":     reward,
            }

        except Exception as e:
            print(f"[EnvWorker] Episode {adapter_id} failed: {e}")
            return {
                "adapter_id": adapter_id,
                "kl":         0.0,
                "score":      0.0,
                "reward":     0.0,
                "error":      str(e),
            }

        finally:
            self._restore_base_weights()

    def run_episodes_serial(self, episodes: list[dict]) -> list[dict]:
        return [self.run_episode(**ep) for ep in episodes]
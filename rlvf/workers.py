import ray
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import outlines
import outlines.generate
import fcntl

MODEL_PATH  = "/lustre/mlnvme/data/s03skoeh_hpc-rlvf/models/Mistral-7B-v0.3"
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
                torch_dtype=torch.float16,
            ).to(self.device)
            self.model.eval()
            print(f"[EnvWorker] Model loaded, releasing lock.")

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("[EnvWorker] Initializing Outlines regex generator...")
        outlines_model = outlines.from_transformers(self.model, self.tokenizer)
        self.generator = outlines.generate.regex(outlines_model, PVQ_PATTERN)

        print(f"[EnvWorker] Loading base logits from {DATA_PATH}...")
        data = torch.load(DATA_PATH, map_location=self.device)
        self.base_logits  = data["logits"]
        self.domain_texts = data["texts"]

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

    def _build_prompt(self, profile: list[float]) -> str:
        return (
            "You are roleplaying as a person.\n"
            "Answer the PVQ-RR questionnaire below as this person.\n\n"
            f"{self.questionnaire_text}\n"
        )

    def _answer_questionnaire(self, profile: list[float]) -> dict:
        prompt = self._build_prompt(profile)
        result = self.generator(prompt, max_new_tokens=200, repetition_penalty=1.3)
        answers = [int(x) for x in result.split(",")]
        return {f"q{i+1}": val for i, val in enumerate(answers)}

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
        z = torch.tensor(z_np, device=self.device, dtype=torch.float32)
        A = {k: torch.tensor(v, dtype=torch.float32) for k, v in A_np.items()}
        B = {k: torch.tensor(v, dtype=torch.float32) for k, v in B_np.items()}

        kl = 0.0  # default in case everything fails

        try:
            self._apply_lora(z, A, B)

            attn = self.model.model.layers[27].self_attn
            delta = (attn.q_proj.weight.data - self._base_weights["27_q"]).abs().max().item()
            print(f"[Diag] max weight delta layer 27 q: {delta:.6f}")
            print(f"[Diag] z mean abs: {z.abs().mean().item():.6f}")

            # KL computed first — always available even if questionnaire fails
            adapted_logits, attention_mask = self._get_adapted_logits()
            kl = self._compute_kl(adapted_logits, attention_mask)

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
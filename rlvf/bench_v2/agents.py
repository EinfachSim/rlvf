from abc import ABC, abstractmethod
import torch
from helpers import load_hn

TARGET_LAYERS = list(range(32))
LAYER_TYPES = ["q", "v"]



def _apply_vera(agent, b, d):
    """h = W0 x + Λ_b·B·Λ_d·A·x."""
    assert not agent._hooks, "adapter hooks already active"
    for li, layer_idx in enumerate(TARGET_LAYERS):
        attn = agent.model.model.layers[layer_idx].self_attn
        for ti, t in enumerate(LAYER_TYPES):
            d_lt = d[li, ti]                                   # (rank,)
            b_lt = b[t][li]                                    # (d_out,)
            A_lt = agent.A[t][li].to(agent.device, torch.float32)     # (rank, d_in)
            B_lt = agent.B[t][li].to(agent.device, torch.float32)     # (d_out, rank)
            proj = attn.q_proj if t == "q" else attn.v_proj

            def hook(module, inputs, output,
                        A_lt=A_lt, B_lt=B_lt, d_lt=d_lt, b_lt=b_lt):
                x = inputs[0].to(torch.float32)
                delta = (((x @ A_lt.T) * d_lt) @ B_lt.T) * b_lt
                return output + delta.to(output.dtype)

            agent._hooks.append(proj.register_forward_hook(hook))


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


class BaseAgent(ABC):

    def __init__(self, model, tokenizer, device="cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        if not self.model or not self.tokenizer:
            raise Exception("Model or Tokenizer not specified!")


        self.system_prompt = None

    def condition(self, profile):
        pass

    def uncondition(self):
        pass

    def tokenize(self, queries):
        """
        Takes in a batch of queries and optionally a system prompt and returns their 
        token ids as a list of lists of ints
        """
        prompt_ids = []
        for query in queries:
            if self.system_prompt:
                query = self.system_prompt + "\n\n" + query
            text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": query}],
                add_generation_prompt=True,
                tokenize=False,
            )
            ids = self.tokenizer(text, add_special_tokens=False).input_ids

            prompt_ids.append(ids)
        return prompt_ids

    def pad(self, input_ids_list):
        max_len = max(len(s) for s in input_ids_list)
        input_ids = torch.full((len(input_ids_list), max_len), self.tokenizer.pad_token_id, dtype=torch.long)
        attn = torch.zeros_like(input_ids)
        for i, s in enumerate(input_ids_list):
            input_ids[i, max_len - len(s):] = torch.tensor(s)
            attn[i, max_len - len(s):] = 1
        return input_ids, attn

    def get_logits(self, queries, profile):
        """
        receives a batch of queries and returns logits of the answer tokens, conditioned on the profile
        """
        print(profile)
        self.condition(profile)
        input_ids = self.tokenize(queries)
        input_ids, attn_mask = self.pad(input_ids)
        logits = self.model(input_ids=input_ids.to(self.device), attention_mask=attn_mask.to(self.device), use_cache=False, logits_to_keep=1).logits
        self.uncondition()
        return logits
    


class HNAgent(BaseAgent):
    
    def __init__(self, hn_path, name, shared_kwargs):
        super().__init__(**shared_kwargs)
        self.hn, self.A, self.B = load_hn(hn_path, self.device)
        self.name = name
        self._hooks = []

    @torch.no_grad()
    def condition(self, profile):
        x = profile.to(self.device).unsqueeze(0)
        b, d = self.hn.act(x)
        _apply_vera(self, b, d)

    def uncondition(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

class PromptedAgent(BaseAgent):

    def __init__(self, shared_kwargs):
        super().__init__(**shared_kwargs)
        self.name = "prompted"

    def condition(self, profile):
        self.system_prompt = render_profile_prompt(profile)


    
    
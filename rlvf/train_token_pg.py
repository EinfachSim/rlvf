"""
train_token_pg.py

Token-level policy gradient — "tokens as actions", RLHF-shaped, one epoch per
batch (ratio ≡ 1, so no clipping machinery is needed; adding true multi-epoch
PPO would require one extra worker round trip per epoch to recompute per-token
log-probs — see design_notes_rlvf.md §3).

Per episode, the worker: samples the 57 answers from the renormalized 6-way
digit distributions of the ADAPTED model; computes the TRUE DISCRETE score;
computes exact leave-one-out counterfactual advantages per item (closed-form
baseline — no critic); backprops
    L = Σ_i [ -logπ_i(a_i)·adv_i − ent_coef·H_i ]
        + tok_kl_weight · mean_i KL_i(adapted‖base @ answer pos)
        + kl_weight · KL_domain
through the frozen LLM to (b, d) and ships the gradients back. The head node
applies the identical VJP surrogate as the pathwise trainer.

Recommended use: initialize from a pathwise checkpoint (SFT-analog), then run
this on the discrete metric (RL stage).
"""

from pathlib import Path
import ray
import torch
import wandb

from rlvf.env import RLVFEnv
from rlvf.model import HyperNetwork
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num_workers", type=int, default=2)
parser.add_argument("--episodes_per_worker", type=int, default=16)
parser.add_argument("--init_from", type=str, default="",
                    help="path to a pathwise checkpoint to warm-start from")
args = parser.parse_args()

NUM_WORKERS         = args.num_workers
EPISODES_PER_WORKER = args.episodes_per_worker

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE          = "cuda:0"
LUSTRE          = "/lustre/mlnvme/data/s03skoeh_hpc-rlvf"
CHECKPOINT_DIR  = f"{LUSTRE}/checkpoints_tokenpg_v1"

NUM_LAYERS      = 32
LAYER_TYPES     = ["q", "v"]
DIMS            = {"q": [4096, 4096], "v": [1024, 4096]}
PROFILE_DIM     = 19
RANK            = 8

LR              = 3e-5          # PG is noisier than pathwise — smaller lr
GRAD_CLIP       = 1.0
BATCH_SIZE      = 64
KL_WEIGHT       = 0.1           # domain-text KL weight
TOK_KL_WEIGHT   = 0.05          # per-answer-position KL weight
ENT_COEF        = 0.0
TEMPERATURE     = 1.0

TOTAL_STEPS     = 1000
EVAL_EVERY      = 10
SAVE_EVERY      = 10

# ── Init ──────────────────────────────────────────────────────────────────────
print("Initializing Weights & Biases...")
wandb.init(
    project="rlvf-pvq-alignment",
    name="mistral-7b-hypernetwork-tokenpg-v1",
    config={
        "num_layers": NUM_LAYERS, "layer_types": LAYER_TYPES, "rank": RANK,
        "lr": LR, "grad_clip": GRAD_CLIP, "batch_size": BATCH_SIZE,
        "kl_weight": KL_WEIGHT, "tok_kl_weight": TOK_KL_WEIGHT,
        "ent_coef": ENT_COEF, "temperature": TEMPERATURE,
        "total_steps": TOTAL_STEPS, "trainer": "token_pg",
    },
)

ray.init(address="auto")
print(f"Ray cluster resources: {ray.cluster_resources()}")

print("Initialising HyperNetwork...")
policy = HyperNetwork(
    num_layers=NUM_LAYERS, layer_types=LAYER_TYPES, dims=DIMS,
    profile_dim=PROFILE_DIM, rank=RANK,
).to(DEVICE)
policy.train()

optimizer = torch.optim.Adam(policy.parameters(), lr=LR)

if args.init_from:
    ckpt = torch.load(args.init_from, map_location=DEVICE)
    policy.load_state_dict(ckpt["model"])
    print(f"[train] Warm-started from {args.init_from} (step {ckpt['step']})")

print("Spawning environment workers...")
env = RLVFEnv(
    num_workers=NUM_WORKERS, episodes_per_worker=EPISODES_PER_WORKER,
    kl_weight=KL_WEIGHT, batch_size=BATCH_SIZE,
    layer_types=len(LAYER_TYPES), rank=RANK, num_layers=NUM_LAYERS,
)
print(f"Batch size: {env.batch_size}")

Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)

def save_checkpoint(step: int):
    path = f"{CHECKPOINT_DIR}/hn_step_{step:05d}.pt"
    torch.save({"step": step, "model": policy.state_dict(),
                "optimizer": optimizer.state_dict()}, path)
    print(f"[train] Checkpoint saved to {path}")

start_step = 0
ckpts = sorted(Path(CHECKPOINT_DIR).glob("hn_step_*.pt"))
if ckpts:
    ckpt = torch.load(str(ckpts[-1]), map_location=DEVICE)
    policy.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    start_step = ckpt["step"]
    print(f"[train] Resumed from {ckpts[-1]} (step {start_step})")

GRAD_KWARGS = {
    "tok_kl_weight": TOK_KL_WEIGHT,
    "ent_coef": ENT_COEF,
    "temperature": TEMPERATURE,
}

print(f"\nStarting token-PG training from step {start_step}...")

for step in range(start_step, TOTAL_STEPS):

    states = env.get_observation_batch(env.batch_size).to(DEVICE)

    with torch.no_grad():
        b_ship, d_ship = policy.act(states)
    A, B = policy.A, policy.B

    res = env.step_batch_grad(
        action_batch=({k: v.cpu() for k, v in b_ship.items()}, d_ship.cpu()),
        states=states.cpu(), A=A, B=B, mode="token_pg",
        grad_kwargs=GRAD_KWARGS,
    )
    ok = res["ok"]
    n_ok = int(ok.sum().item())
    if n_ok == 0:
        print(f"[train] step {step:04d} | ALL episodes failed — skipping update")
        continue
    g_d = res["grad_d"].to(DEVICE)
    g_b = {k: v.to(DEVICE) for k, v in res["grad_b"].items()}

    b2, d2 = policy.act(states)
    surrogate = (d2 * g_d.detach()).sum()
    for k in g_b:
        surrogate = surrogate + (b2[k] * g_b[k].detach()).sum()
    surrogate = surrogate / n_ok

    optimizer.zero_grad()
    surrogate.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), GRAD_CLIP)
    optimizer.step()

    rewards = res["rewards"]
    kl_tok = [r.get("kl_token", float("nan")) for r in res["results"]]
    ent    = [r.get("entropy", float("nan")) for r in res["results"]]
    adv_sd = [r.get("adv_std", float("nan")) for r in res["results"]]
    kl_tok_mean = float(torch.tensor(kl_tok)[ok].nanmean().item())
    ent_mean    = float(torch.tensor(ent)[ok].nanmean().item())
    adv_sd_mean = float(torch.tensor(adv_sd)[ok].nanmean().item())

    log_dict = {
        "train/reward_mean": rewards[ok].mean().item(),
        "train/reward_std":  rewards[ok].std().item(),
        "train/score_mean":  float(env.last_scores[ok].mean().item()),
        "train/kl_mean":     float(env.last_kls[ok].mean().item()),
        "train/kl_token_mean": kl_tok_mean,
        "train/entropy_mean":  ent_mean,
        "train/adv_std_mean":  adv_sd_mean,
        "train/digit_mass":  float(env.last_digit_mass[ok].mean().item()),
        "train/errors":      env.last_errors,
        "train/env_grad_norm_mean": res["env_grad_norms"][ok].mean().item(),
        "train/hn_grad_norm": grad_norm.item(),
        "step": step,
    }
    wandb.log(log_dict, step=step)
    print(
        f"[train] step {step:04d} | "
        f"reward: {log_dict['train/reward_mean']:+.4f} | "
        f"score(disc): {log_dict['train/score_mean']:+.4f} | "
        f"kl_tok: {kl_tok_mean:.4f} | H: {ent_mean:.3f} | "
        f"adv_sd: {adv_sd_mean:.3f} | hn_g: {grad_norm.item():.4f} | "
        f"err: {env.last_errors}"
    )

    if step % EVAL_EVERY == 0:
        eval_states = env.eval_profiles.to(DEVICE)
        with torch.no_grad():
            eval_b, eval_d = policy.act(eval_states)
        eval_metrics = env.eval_batch(
            action_batch=({k: v.cpu() for k, v in eval_b.items()},
                          eval_d.cpu()),
            A=A, B=B,
        )
        wandb.log({
            "eval/reward_mean": eval_metrics["eval_reward_mean"],
            "eval/reward_std":  eval_metrics["eval_reward_std"],
            "eval/score_mean":  eval_metrics["eval_score_mean"],
            "eval/kl_mean":     eval_metrics["eval_kl_mean"],
            "eval/digit_mass":  eval_metrics["eval_digit_mass"],
        }, step=step)
        print(f"[eval]  step {step:04d} | "
              f"reward: {eval_metrics['eval_reward_mean']:+.4f} ± "
              f"{eval_metrics['eval_reward_std']:.4f} | "
              f"score: {eval_metrics['eval_score_mean']:+.4f}")

    if step % SAVE_EVERY == 0 and step > 0:
        save_checkpoint(step)

save_checkpoint(TOTAL_STEPS)
print("Training complete.")
wandb.finish()
ray.shutdown()
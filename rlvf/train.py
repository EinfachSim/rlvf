"""
train.py

Main training loop. Runs on the SLURM head node (has GPU for HyperNetwork).
Ray workers on the other node handle environment computation.
"""

from pathlib import Path
import ray
import torch
import wandb

from rlvf.env import RLVFEnv
from rlvf.model import HyperNetwork
from rlvf.ppo import PPO
import argparse

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--num_workers", type=int, default=2)
parser.add_argument("--episodes_per_worker", type=int, default=16)
args = parser.parse_args()

NUM_WORKERS         = args.num_workers
EPISODES_PER_WORKER = args.episodes_per_worker

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE          = "cuda:0"
LUSTRE          = "/lustre/mlnvme/data/s03skoeh_hpc-rlvf"
CHECKPOINT_DIR  = f"{LUSTRE}/checkpoints"

# HyperNetwork
NUM_LAYERS      = 32
LAYER_TYPES     = ["q", "v"]
DIMS            = {"q": [4096, 4096], "v": [1024, 4096]}
PROFILE_DIM     = 19
RANK            = 8

# PPO
LR              = 5e-5
CLIP_RATIO      = 0.15
VF_COEF         = 0.5
ENT_COEF        = 0.0
TARGET_KL       = 0.05
PPO_EPOCHS      = 10
BATCH_SIZE      = 64

# Env
KL_WEIGHT       = 0.1

# Training
TOTAL_STEPS     = 1000
EVAL_EVERY      = 10
SAVE_EVERY      = 10
LOG_EVERY       = 1

# ── Init WandB ────────────────────────────────────────────────────────────────
print("Initializing Weights & Biases...")
wandb.init(
    project="rlvf-pvq-alignment",
    name="mistral-7b-hypernetwork-ppo",
    config={
        "num_layers":           NUM_LAYERS,
        "layer_types":          LAYER_TYPES,
        "rank":                 RANK,
        "lr":                   LR,
        "clip_ratio":           CLIP_RATIO,
        "vf_coef":              VF_COEF,
        "ent_coef":             ENT_COEF,
        "target_kl":            TARGET_KL,
        "ppo_epochs":           PPO_EPOCHS,
        "num_workers":          NUM_WORKERS,
        "episodes_per_worker":  EPISODES_PER_WORKER,
        "kl_weight":            KL_WEIGHT,
        "total_steps":          TOTAL_STEPS,
    }
)

# ── Init Ray ──────────────────────────────────────────────────────────────────
ray.init(address="auto")
print(f"Ray cluster resources: {ray.cluster_resources()}")

# ── Init HyperNetwork on GPU ──────────────────────────────────────────────────
print("Initialising HyperNetwork...")
policy = HyperNetwork(
    num_layers  = NUM_LAYERS,
    layer_types = LAYER_TYPES,
    dims        = DIMS,
    profile_dim = PROFILE_DIM,
    rank        = RANK,
).to(DEVICE)

#Value head warm start
with torch.no_grad():
    policy.value_head[-1].bias.fill_(-2.0)

# ── Init PPO ──────────────────────────────────────────────────────────────────
ppo = PPO(
    policy      = policy,
    num_iter    = PPO_EPOCHS,
    clip_ratio  = CLIP_RATIO,
    lr          = LR,
    vf_coef     = VF_COEF,
    ent_coef    = ENT_COEF,
    target_kl   = TARGET_KL,
)

# ── Init Env ──────────────────────────────────────────────────────────────────
print("Spawning environment workers...")
env = RLVFEnv(
    num_workers         = NUM_WORKERS,
    episodes_per_worker = EPISODES_PER_WORKER,
    kl_weight           = KL_WEIGHT,
    batch_size          = BATCH_SIZE,
    layer_types         = len(LAYER_TYPES),
    rank                = RANK,
    num_layers          = NUM_LAYERS,
)
print(f"Batch size: {env.batch_size}")

# ── Checkpoint helpers ────────────────────────────────────────────────────────
Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)

def save_checkpoint(step: int):
    path = f"{CHECKPOINT_DIR}/hn_step_{step:05d}.pt"
    torch.save({
        "step":      step,
        "model":     policy.state_dict(),
        "optimizer": ppo.optimizer.state_dict(),
    }, path)
    print(f"[train] Checkpoint saved to {path}")

def load_checkpoint(path: str) -> int:
    ckpt = torch.load(path, map_location=DEVICE)
    policy.load_state_dict(ckpt["model"])
    ppo.optimizer.load_state_dict(ckpt["optimizer"])
    print(f"[train] Resumed from {path} (step {ckpt['step']})")
    return ckpt["step"]

# ── Resume from latest checkpoint if exists ───────────────────────────────────
start_step = 0
ckpts = sorted(Path(CHECKPOINT_DIR).glob("hn_step_*.pt"))
if ckpts:
    start_step = load_checkpoint(str(ckpts[-1]))

# ── Training loop ─────────────────────────────────────────────────────────────
print(f"\nStarting training from step {start_step}...")

for step in range(start_step, TOTAL_STEPS):

    # 1. Sample fresh profiles for this batch
    states = env.get_observation_batch(env.batch_size).to(DEVICE)  # (B, 19)

    # 2. Sample actions from policy
    with torch.no_grad():
        actions, log_probs, A, B = policy.get_action_and_logprob(states)
        actions   = actions.detach()
        log_probs = log_probs.detach()

    # 3. Fan out to environment workers
    rewards = env.step_batch(
        action_batch = actions.cpu(),
        states       = states.cpu(),
        A            = A,
        B            = B,
    ).to(DEVICE)  # (B,)

    # 4. PPO update
    batch = (
        states.detach(),
        actions.detach(),
        rewards.detach(),
        log_probs.detach(),
    )
    ppo_info = ppo.update(batch)

    # 5. Logging
    if step % LOG_EVERY == 0:
        log_dict = {
            "train/reward_mean": rewards.mean().item(),
            "train/reward_std":  rewards.std().item(),
            "train/reward_min":  rewards.min().item(),
            "train/reward_max":  rewards.max().item(),
            # Action diagnostics
            "diag/z_mean":       actions.mean().item(),
            "diag/z_std":        actions.std().item(),
            "diag/logp_old":     log_probs.mean().item(),
            "step": step,
        }

        if hasattr(env, "last_scores") and env.last_scores is not None:
            log_dict["train/score_mean"] = float(env.last_scores.mean().item())
        if hasattr(env, "last_kls") and env.last_kls is not None:
            log_dict["train/kl_mean"] = float(env.last_kls.mean().item())

        if isinstance(ppo_info, dict):
            for k, v in ppo_info.items():
                if isinstance(v, torch.Tensor):
                    v = v.item()
                log_dict[f"ppo/{k}"] = v
        # Log A and B gradient norms and weight norms
        with torch.no_grad():
            for t in LAYER_TYPES:
                A = policy.A[t]
                B = policy.B[t]
                        
                # Weight norms — if these don't change across steps, A/B are frozen
                log_dict[f"diag/A_{t}_norm"] = A.norm().item()
                log_dict[f"diag/B_{t}_norm"] = B.norm().item()
                        
                # Gradient norms — if these are 0, A/B receive no gradient at all
                if A.grad is not None:
                    log_dict[f"diag/A_{t}_grad_norm"] = A.grad.norm().item()
                    log_dict[f"diag/B_{t}_grad_norm"] = B.grad.norm().item()
                else:
                    log_dict[f"diag/A_{t}_grad_norm"] = 0.0
                    log_dict[f"diag/B_{t}_grad_norm"] = 0.0

        wandb.log(log_dict, step=step)

        print(
            f"[train] step {step:04d} | "
            f"reward: {rewards.mean().item():+.4f} | "
            f"logp_old: {log_probs.mean().item():.4f} | "
            f"logp_new: {ppo_info.get('logp_new_mean', 0):.4f} | "
            f"ratio: {ppo_info.get('ratio_mean', 0):.4f} | "
            f"grad_norm: {ppo_info.get('grad_norm', 0):.4f} | "
            f"z_mean: {actions.mean().item():.4f}"
        )

    # 6. Eval
    if step % EVAL_EVERY == 0:
        eval_states = env.eval_profiles.to(DEVICE)
        with torch.no_grad():
            eval_actions, _, eval_A, eval_B = policy.get_action_and_logprob(eval_states)
            eval_actions = eval_actions.detach()
        eval_metrics = env.eval_batch(
            action_batch = eval_actions.cpu(),
            A            = eval_A,
            B            = eval_B,
        )

        eval_log_dict = {
            "eval/reward_mean": eval_metrics.get("eval_reward_mean", 0.0),
            "eval/reward_std":  eval_metrics.get("eval_reward_std", 0.0),
        }
        if "eval_score_mean" in eval_metrics:
            eval_log_dict["eval/score_mean"] = eval_metrics["eval_score_mean"]
        if "eval_kl_mean" in eval_metrics:
            eval_log_dict["eval/kl_mean"] = eval_metrics["eval_kl_mean"]

        wandb.log(eval_log_dict, step=step)

        print(
            f"[eval]  step {step:04d} | "
            f"reward: {eval_metrics['eval_reward_mean']:+.4f} ± "
            f"{eval_metrics['eval_reward_std']:.4f}"
        )

    # 7. Checkpoint
    if step % SAVE_EVERY == 0 and step > 0:
        save_checkpoint(step)

# Final checkpoint
save_checkpoint(TOTAL_STEPS)
print("Training complete. Closing WandB run...")
wandb.finish()
ray.shutdown()
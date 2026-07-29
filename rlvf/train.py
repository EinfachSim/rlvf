"""
train.py

Main training loop. Runs on the SLURM head node (has GPU for HyperNetwork).
Ray workers on the other node handle environment computation.
"""

import ray
import torch
from pathlib import Path

from rlvf.model import HyperNetwork
from rlvf.ppo import PPO
from rlvf.env import RLVFEnv

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE          = "cuda:0"
LUSTRE          = "/lustre/mlnvme/data/s03skoeh_hpc-rlvf"
CHECKPOINT_DIR  = f"{LUSTRE}/checkpoints"

# HyperNetwork
NUM_LAYERS      = 5
LAYER_TYPES     = ["q", "v"]
DIMS            = {"q": [4096, 4096], "v": [1024, 4096]}
PROFILE_DIM     = 19
RANK            = 8

# PPO
LR              = 1e-3
CLIP_RATIO      = 0.2
VF_COEF         = 0.5
ENT_COEF        = 0.01
TARGET_KL       = 0.01
PPO_EPOCHS      = 2

# Env
NUM_WORKERS         = 8
EPISODES_PER_WORKER = 8   # 8 × 8 = 64 total per batch
KL_WEIGHT           = 0.1

# Training
TOTAL_STEPS     = 1000
EVAL_EVERY      = 10
SAVE_EVERY      = 50
LOG_EVERY       = 1

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
)
print(f"Batch size: {env.batch_size}")

# ── Checkpoint helpers ────────────────────────────────────────────────────────
Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)

def save_checkpoint(step: int):
    path = f"{CHECKPOINT_DIR}/hn_step_{step:05d}.pt"
    torch.save({
        "step":       step,
        "model":      policy.state_dict(),
        "optimizer":  ppo.optimizer.state_dict(),
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

    # 2. Sample actions from policy — detach everything before handing to PPO
    #    PPO recomputes log_probs internally via _evaluate for the gradient flow
    with torch.no_grad():
        actions, log_probs, A, B = policy.get_action_and_logprob(states)
        actions   = actions.detach()
        log_probs = log_probs.detach()

    # 3. Fan out to environment workers — collect rewards
    rewards = env.step_batch(
        action_batch = actions.cpu(),
        states       = states.cpu(),
        A            = A,
        B            = B,
    ).to(DEVICE)  # (B,)

    # 4. PPO update — pass fully detached batch
    batch = (
        states.detach(),
        actions.detach(),
        rewards.detach(),
        log_probs.detach(),
    )
    ppo.update(batch)

    # 5. Logging
    if step % LOG_EVERY == 0:
        print(
            f"[train] step {step:04d} | "
            f"reward: {rewards.mean().item():+.4f} | "
            f"reward_std: {rewards.std().item():.4f}"
        )

    # 6. Eval
    if step % EVAL_EVERY == 0:
        eval_states = env.eval_profiles.to(DEVICE)  # fixed (32, 19)
        with torch.no_grad():
            eval_actions, _, eval_A, eval_B = policy.get_action_and_logprob(eval_states)
            eval_actions = eval_actions.detach()
        eval_metrics = env.eval_batch(
            action_batch = eval_actions.cpu(),
            A            = eval_A,
            B            = eval_B,
        )
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
print("Training complete.")
ray.shutdown()
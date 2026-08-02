"""
train.py

Main training loop. Runs on the SLURM head node (has GPU for HyperNetwork).
Ray workers on the other node handle environment computation.

CHANGES vs previous version
---------------------------
1. Paper-faithful VeRA actions (b, d): b per OUTPUT dim per layer/type,
   d per rank. Total action dim = 32*(4096+1024) + 512 = 164,352.
2. Eval uses deterministic=True (policy means). Previously eval sampled
   actions, so it measured policy+exploration-noise, not the policy.
3. train/score_mean, train/kl_mean, digit mass, ΔW norm, and error counts are
   now logged for real (env.last_* is populated now — the old hasattr branch
   never fired, which is why the CSV had no score/KL decomposition).
4. log_std is logged every step (would have caught the -2 vs -4 init mismatch
   immediately).
5. target_kl = 0.02 and clip_ratio = 0.2 on TRUE joint ratios/KL. The old
   values were calibrated (accidentally) to per-dim-normalized quantities.
6. CHECKPOINT_DIR bumped to checkpoints_v3: the architecture changed again
   (b heads per type, separate log_stds), so auto-resume from v1/v2
   checkpoints would crash or silently mis-load.
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
CHECKPOINT_DIR  = f"{LUSTRE}/checkpoints_v3"   # v3: VeRA (b,d) — incompatible with v1/v2 ckpts

# HyperNetwork
NUM_LAYERS      = 32
LAYER_TYPES     = ["q", "v"]
DIMS            = {"q": [4096, 4096], "v": [1024, 4096]}
PROFILE_DIM     = 19
RANK            = 8

# PPO — calibrated for TRUE joint ratios / KL (nats).
# TARGET_KL is a TOTAL KL over the FULL action (now 164,352 dims with VeRA
# b in R^{d_out}). KL scales with action dim: even minute per-dim policy
# movement produces large total KL, so expect update_iters to sit at 1-2 —
# PPO then behaves like REINFORCE-with-baseline, which is fine and safe.
# TARGET_KL below = ~6e-6 nats/dim, deliberately loose in total terms.
# NOTE the early stop can only prevent ADDITIONAL epochs; the size of the
# FIRST gradient step of each update is set by LR alone. Do not raise LR
# without rechecking ppo/approx_kl.
LR              = 5e-5
CLIP_RATIO      = 0.2
VF_COEF         = 0.5
ENT_COEF        = 0.0
TARGET_KL       = 1.0
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
    name="mistral-7b-hypernetwork-ppo-v3-vera",
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

# Value head warm start
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
        (action_b, action_d), log_probs, A, B = policy.get_action_and_logprob(states)
        action_b  = {k: v.detach() for k, v in action_b.items()}
        action_d  = action_d.detach()
        log_probs = log_probs.detach()

    # 3. Fan out to environment workers
    rewards = env.step_batch(
        action_batch = ({k: v.cpu() for k, v in action_b.items()},
                        action_d.cpu()),
        states       = states.cpu(),
        A            = A,
        B            = B,
    ).to(DEVICE)  # (B,)

    # 4. PPO update
    batch = (
        states.detach(),
        (action_b, action_d),
        rewards.detach(),
        log_probs,
    )
    ppo_info = ppo.update(batch)

    # 5. Logging
    if step % LOG_EVERY == 0:
        with torch.no_grad():
            log_std_mean = policy.log_std_mean().item()
            b_flat = torch.cat([v.flatten() for v in action_b.values()])

        log_dict = {
            "train/reward_mean": rewards.mean().item(),
            "train/reward_std":  rewards.std().item(),
            "train/reward_min":  rewards.min().item(),
            "train/reward_max":  rewards.max().item(),
            # Reward decomposition — env.last_* is populated now
            "train/score_mean":  float(env.last_scores.mean().item()),
            "train/kl_mean":     float(env.last_kls.mean().item()),
            "train/digit_mass":  float(env.last_digit_mass.mean().item()),
            "train/delta_w_fro": float(env.last_delta_fro.nanmean().item()),
            "train/errors":      env.last_errors,
            # Action diagnostics
            "diag/d_mean":       action_d.mean().item(),
            "diag/d_std":        action_d.std().item(),
            "diag/d_abs_mean":   action_d.abs().mean().item(),
            "diag/b_mean":       b_flat.mean().item(),
            "diag/b_std":        b_flat.std().item(),
            "diag/b_abs_mean":   b_flat.abs().mean().item(),
            "diag/log_std":      log_std_mean,
            "diag/logp_old":     log_probs.mean().item(),
            "step": step,
        }

        if isinstance(ppo_info, dict):
            for k, v in ppo_info.items():
                if isinstance(v, torch.Tensor):
                    v = v.item()
                log_dict[f"ppo/{k}"] = v

        wandb.log(log_dict, step=step)

        print(
            f"[train] step {step:04d} | "
            f"reward: {rewards.mean().item():+.4f} | "
            f"score: {env.last_scores.mean().item():+.4f} | "
            f"kl: {env.last_kls.mean().item():.4f} | "
            f"approx_kl: {ppo_info.get('approx_kl', 0):.4f} | "
            f"iters: {ppo_info.get('update_iters', 0)} | "
            f"ratio: {ppo_info.get('ratio_mean', 0):.4f} | "
            f"grad_norm: {ppo_info.get('grad_norm', 0):.4f} | "
            f"log_std: {log_std_mean:.3f}"
        )

    # 6. Eval — deterministic (policy means), so it measures the policy itself
    if step % EVAL_EVERY == 0:
        eval_states = env.eval_profiles.to(DEVICE)
        with torch.no_grad():
            (eval_b, eval_d), _, eval_A, eval_B = policy.get_action_and_logprob(
                eval_states, deterministic=True
            )
        eval_metrics = env.eval_batch(
            action_batch = ({k: v.cpu() for k, v in eval_b.items()},
                            eval_d.cpu()),
            A            = eval_A,
            B            = eval_B,
        )

        eval_log_dict = {
            "eval/reward_mean": eval_metrics.get("eval_reward_mean", 0.0),
            "eval/reward_std":  eval_metrics.get("eval_reward_std", 0.0),
            "eval/score_mean":  eval_metrics.get("eval_score_mean", 0.0),
            "eval/kl_mean":     eval_metrics.get("eval_kl_mean", 0.0),
            "eval/digit_mass":  eval_metrics.get("eval_digit_mass", 0.0),
        }
        wandb.log(eval_log_dict, step=step)

        print(
            f"[eval]  step {step:04d} | "
            f"reward: {eval_metrics['eval_reward_mean']:+.4f} ± "
            f"{eval_metrics['eval_reward_std']:.4f} | "
            f"score: {eval_metrics['eval_score_mean']:+.4f} | "
            f"kl: {eval_metrics['eval_kl_mean']:.4f}"
        )

    # 7. Checkpoint
    if step % SAVE_EVERY == 0 and step > 0:
        save_checkpoint(step)

# Final checkpoint
save_checkpoint(TOTAL_STEPS)
print("Training complete. Closing WandB run...")
wandb.finish()
ray.shutdown()
"""
train_pathwise.py

Pathwise (fully differentiable) trainer for the hypernetwork.
Head node: hypernetwork forward (deterministic) -> ship (b, d) -> workers
return per-episode dL/db, dL/dd (L = -(smooth score) + kl_weight*KL) ->
head recomputes the cheap hypernetwork forward with grad and backprops the
VJP surrogate  mean_i [ <b_i, g_b_i> + <d_i, g_d_i> ]  -> Adam step.
This equals end-to-end backprop with the chain rule split at (b, d); see
design_notes_rlvf.md §1 and §3. No sampling, no ratios, no critic.
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
args = parser.parse_args()

NUM_WORKERS         = args.num_workers
EPISODES_PER_WORKER = args.episodes_per_worker

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE          = "cuda:0"
LUSTRE          = "/lustre/mlnvme/data/s03skoeh_hpc-rlvf"
CHECKPOINT_DIR  = f"{LUSTRE}/checkpoints_pathwise_v1"

NUM_LAYERS      = 32
LAYER_TYPES     = ["q", "v"]
DIMS            = {"q": [4096, 4096], "v": [1024, 4096]}
PROFILE_DIM     = 19
RANK            = 8

LR              = 1e-4          # supervised-style; exact gradients
GRAD_CLIP       = 1.0
BATCH_SIZE      = 64
KL_WEIGHT       = 0.1

TOTAL_STEPS     = 1000
EVAL_EVERY      = 10
SAVE_EVERY      = 10

# ── Init ──────────────────────────────────────────────────────────────────────
print("Initializing Weights & Biases...")
wandb.init(
    project="rlvf-pvq-alignment",
    name="mistral-7b-hypernetwork-pathwise-v1",
    config={
        "num_layers": NUM_LAYERS, "layer_types": LAYER_TYPES, "rank": RANK,
        "lr": LR, "grad_clip": GRAD_CLIP, "batch_size": BATCH_SIZE,
        "kl_weight": KL_WEIGHT, "total_steps": TOTAL_STEPS,
        "trainer": "pathwise",
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

def load_checkpoint(path: str) -> int:
    ckpt = torch.load(path, map_location=DEVICE)
    policy.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    print(f"[train] Resumed from {path} (step {ckpt['step']})")
    return ckpt["step"]

start_step = 0
ckpts = sorted(Path(CHECKPOINT_DIR).glob("hn_step_*.pt"))
if ckpts:
    start_step = load_checkpoint(str(ckpts[-1]))

print(f"\nStarting pathwise training from step {start_step}...")

for step in range(start_step, TOTAL_STEPS):

    # 1. Fresh profiles
    states = env.get_observation_batch(env.batch_size).to(DEVICE)   # (B, 19)

    # 2. Deterministic actions to ship (no grad needed for the rollout copy)
    with torch.no_grad():
        b_ship, d_ship = policy.act(states)
    A, B = policy.A, policy.B

    # 3. Workers compute rewards AND dL/d(b, d)
    res = env.step_batch_grad(
        action_batch=({k: v.cpu() for k, v in b_ship.items()}, d_ship.cpu()),
        states=states.cpu(), A=A, B=B, mode="pathwise",
    )
    ok = res["ok"]
    n_ok = int(ok.sum().item())
    if n_ok == 0:
        print(f"[train] step {step:04d} | ALL episodes failed — skipping update")
        continue
    g_d = res["grad_d"].to(DEVICE)                    # (B, L*T, r)
    g_b = {k: v.to(DEVICE) for k, v in res["grad_b"].items()}

    # 4. VJP surrogate — autograd sums per-episode VJPs into policy grads.
    #    Failed episodes contribute zero grads by construction; divide by n_ok.
    b2, d2 = policy.act(states)                       # WITH grad
    surrogate = (d2 * g_d.detach()).sum()
    for k in g_b:
        surrogate = surrogate + (b2[k] * g_b[k].detach()).sum()
    surrogate = surrogate / n_ok

    optimizer.zero_grad()
    surrogate.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), GRAD_CLIP)
    optimizer.step()

    # 5. Logging
    rewards = res["rewards"]
    log_dict = {
        "train/reward_mean": rewards[ok].mean().item(),
        "train/reward_std":  rewards[ok].std().item(),
        "train/score_mean":  float(env.last_scores[ok].mean().item()),
        "train/kl_mean":     float(env.last_kls[ok].mean().item()),
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
        f"score: {log_dict['train/score_mean']:+.4f} | "
        f"kl: {log_dict['train/kl_mean']:.4f} | "
        f"env_g: {log_dict['train/env_grad_norm_mean']:.4f} | "
        f"hn_g: {log_dict['train/hn_grad_norm']:.4f} | "
        f"err: {env.last_errors}"
    )

    # 6. Eval — deterministic reward-only episodes on the fixed eval profiles
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
              f"score: {eval_metrics['eval_score_mean']:+.4f} | "
              f"kl: {eval_metrics['eval_kl_mean']:.4f}")

    if step % SAVE_EVERY == 0 and step > 0:
        save_checkpoint(step)

save_checkpoint(TOTAL_STEPS)
print("Training complete.")
wandb.finish()
ray.shutdown()
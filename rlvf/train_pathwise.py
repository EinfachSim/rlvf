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
                    help="optional checkpoint to warm-start the HN weights from "
                         "(optimizer state is NOT restored, step counter starts at 0). "
                         "Use a pre-saturation checkpoint, e.g. v4 step 20-25.")
args = parser.parse_args()

NUM_WORKERS         = args.num_workers
EPISODES_PER_WORKER = args.episodes_per_worker

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE          = "cuda:0"
LUSTRE          = "/lustre/mlnvme/data/s03skoeh_hpc-rlvf"
CHECKPOINT_DIR  = f"{LUSTRE}/checkpoints_pathwise_v5"   # new dir: v4 is saturated

NUM_LAYERS      = 32
LAYER_TYPES     = ["q", "v"]
DIMS            = {"q": [4096, 4096], "v": [1024, 4096]}
PROFILE_DIM     = 19
RANK            = 8

LR              = 3e-4
GRAD_CLIP       = 1.0
BATCH_SIZE      = 64
KL_WEIGHT       = 0.1
MASS_COEF       = 0.1
# Entropy floor on the 6-digit answer distributions (see workers.run_episode_grad).
# Keeps the expected-value surrogate's gradient alive. Watch train/pmax: it
# should stay clearly below 1 (e.g. < 0.9) while the score improves.
ENT_FLOOR_COEF  = 0.1
H_MIN           = 0.5     # nats; ln(6) = 1.79 is the maximum

TOTAL_STEPS     = 1000
EVAL_EVERY      = 5
SAVE_EVERY      = 5

# ── Init ──────────────────────────────────────────────────────────────────────
print("Initializing Weights & Biases...")
wandb.init(
    project="rlvf-pvq-alignment",
    name="mistral-7b-hypernetwork-pathwise-v5-entfloor",
    config={
        "num_layers": NUM_LAYERS, "layer_types": LAYER_TYPES, "rank": RANK,
        "lr": LR, "grad_clip": GRAD_CLIP, "batch_size": BATCH_SIZE,
        "kl_weight": KL_WEIGHT, "mass_coef": MASS_COEF,
        "ent_floor_coef": ENT_FLOOR_COEF, "h_min": H_MIN,
        "total_steps": TOTAL_STEPS, "init_from": args.init_from,
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

Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)

def save_checkpoint(step: int):
    path = f"{CHECKPOINT_DIR}/hn_step_{step:05d}.pt"
    torch.save({"step": step, "model": policy.state_dict(),
                "optimizer": optimizer.state_dict()}, path)
    print(f"[train] Checkpoint saved to {path}")

def load_checkpoint(path: str) -> int:
    ckpt = torch.load(path, map_location=DEVICE)
    result = policy.load_state_dict(ckpt["model"], strict=False)
    print(result.missing_keys)
    print(result.unexpected_keys)
    optimizer.load_state_dict(ckpt["optimizer"])
    print(f"[train] Resumed from {path} (step {ckpt['step']})")
    return ckpt["step"]

start_step = 0
ckpts = sorted(Path(CHECKPOINT_DIR).glob("hn_step_*.pt"))
if ckpts:
    start_step = load_checkpoint(str(ckpts[-1]))
elif args.init_from:
    ckpt = torch.load(args.init_from, map_location=DEVICE)
    policy.load_state_dict(ckpt["model"], strict=True)
    print(f"[train] Warm-started HN weights from {args.init_from} "
          f"(step {ckpt['step']}); fresh optimizer, step counter at 0")

print(f"\nStarting pathwise training from step {start_step}...")

print("Spawning environment workers...")
env = RLVFEnv(
    A=policy.A,
    B=policy.B,
    num_workers=NUM_WORKERS, episodes_per_worker=EPISODES_PER_WORKER,
    kl_weight=KL_WEIGHT, batch_size=BATCH_SIZE,
    layer_types=len(LAYER_TYPES), rank=RANK, num_layers=NUM_LAYERS,
    train_means=None,          # zero-mean: uniform coverage of the profile space
    eval_means="empirical",    # population prior (Schwartz & Cieciuch 2022, Table 5)
)
print(f"Batch size: {env.batch_size}")

_last_step = start_step
try:
    for step in range(start_step, TOTAL_STEPS):
        _last_step = step

        # 1. Fresh profiles
        states = env.get_observation_batch(env.batch_size).to(DEVICE) # (B, 19)

        # 2. Deterministic actions to ship (no grad needed for the rollout copy)
        with torch.no_grad():
            b_ship, d_ship = policy.act(states)

        # 3. Workers compute rewards and dL/d(b, d)
        res = env.step_batch_grad(
            action_batch=({k: v.cpu() for k, v in b_ship.items()}, d_ship.cpu()),
            states=states.cpu(), mode="pathwise",
            grad_kwargs={"mass_coef": MASS_COEF,
                         "ent_floor_coef": ENT_FLOOR_COEF, "h_min": H_MIN},
        )
        ok = res["ok"]
        n_ok = int(ok.sum().item())
        if n_ok == 0:
            print(f"[train] step {step:04d} | ALL episodes failed — skipping update")
            continue
        g_d = res["grad_d"].to(DEVICE) # (B, L*T, r)
        g_b = {k: v.to(DEVICE) for k, v in res["grad_b"].items()}

        # 4. VJP surrogate, autograd sums per-episode VJPs into policy grads.
        #    Failed episodes contribute zero grads by construction; divide by n_ok.
        b2, d2 = policy.act(states) # with grad
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
        mass_pens = torch.tensor([r.get("mass_penalty", float("nan"))
                                  for r in res["results"]])
        ent_pens = torch.tensor([r.get("ent_penalty", float("nan"))
                                 for r in res["results"]])
        score_disc = torch.tensor([r.get("score_disc", float("nan"))
                                   for r in res["results"]])
        log_dict = {
            "train/reward_mean": rewards.mean().item(),
            "train/reward_std":  rewards.std().item(),
            "train/reward_mean_ok": rewards[ok].mean().item(),
            "train/score_mean_ok":  float(env.last_scores[ok].mean().item()),
            "train/score_disc_mean": float(score_disc[ok].nanmean().item()),
            "train/mass_penalty": float(mass_pens[ok].nanmean().item()),
            "train/ent_penalty":  float(ent_pens[ok].nanmean().item()),
            "train/entropy":      float(env.last_entropy[ok].nanmean().item()),
            "train/pmax":         float(env.last_pmax[ok].nanmean().item()),
            "train/score_mean":  float(env.last_scores.mean().item()),
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
            f"reward: {log_dict['train/reward_mean']:+.4f} "
            f"(ok: {log_dict['train/reward_mean_ok']:+.4f}) | "
            f"score: {log_dict['train/score_mean']:+.4f} "
            f"(disc: {log_dict['train/score_disc_mean']:+.4f}) | "
            f"kl: {log_dict['train/kl_mean']:.4f} | "
            f"H: {log_dict['train/entropy']:.3f} | "
            f"pmax: {log_dict['train/pmax']:.3f} | "
            f"env_g: {log_dict['train/env_grad_norm_mean']:.4f} | "
            f"hn_g: {log_dict['train/hn_grad_norm']:.4f} | "
            f"err: {env.last_errors}"
        )

        # 6. Eval, deterministic reward-only episodes on the fixed eval profiles
        if step % EVAL_EVERY == 0:
            eval_states = env.eval_profiles.to(DEVICE)
            with torch.no_grad():
                eval_b, eval_d = policy.act(eval_states)
            eval_metrics = env.eval_batch(
                action_batch=({k: v.cpu() for k, v in eval_b.items()},
                              eval_d.cpu()),
            )
            #zero mean evals 
            eval_states_zero_mean = env.eval_profiles_zero_mean.to(DEVICE)
            with torch.no_grad():
                eval_b, eval_d = policy.act(eval_states_zero_mean)
            eval_metrics_zero_mean = env.eval_batch(
                action_batch=({k: v.cpu() for k, v in eval_b.items()},
                                eval_d.cpu()),
            )
            wandb.log({
                "eval/reward_mean": eval_metrics["eval_reward_mean"],
                "eval/reward_std":  eval_metrics["eval_reward_std"],
                "eval/score_mean":  eval_metrics["eval_score_mean"],
                "eval/score_disc_mean": eval_metrics["eval_score_disc_mean"],
                "eval/kl_mean":     eval_metrics["eval_kl_mean"],
                "eval/digit_mass":  eval_metrics["eval_digit_mass"],
                "eval/entropy":     eval_metrics["eval_entropy"],
                "eval/pmax":        eval_metrics["eval_pmax"],

                "eval_zero_mean/reward_mean": eval_metrics_zero_mean["eval_reward_mean"],
                "eval_zero_mean/reward_std":  eval_metrics_zero_mean["eval_reward_std"],
                "eval_zero_mean/score_mean":  eval_metrics_zero_mean["eval_score_mean"],
                "eval_zero_mean/score_disc_mean": eval_metrics_zero_mean["eval_score_disc_mean"],
                "eval_zero_mean/kl_mean":     eval_metrics_zero_mean["eval_kl_mean"],
                "eval_zero_mean/digit_mass":  eval_metrics_zero_mean["eval_digit_mass"],
                "eval_zero_mean/entropy":     eval_metrics_zero_mean["eval_entropy"],
                "eval_zero_mean/pmax":        eval_metrics_zero_mean["eval_pmax"],
            }, step=step)
            print(f"[eval]  step {step:04d} | "
                  f"reward: {eval_metrics['eval_reward_mean']:+.4f} ± "
                  f"{eval_metrics['eval_reward_std']:.4f} | "
                  f"score: {eval_metrics['eval_score_mean']:+.4f} "
                  f"(disc: {eval_metrics['eval_score_disc_mean']:+.4f}) | "
                  f"kl: {eval_metrics['eval_kl_mean']:.4f} | "
                  f"H: {eval_metrics['eval_entropy']:.3f} | "
                  f"pmax: {eval_metrics['eval_pmax']:.3f}")

        if step % SAVE_EVERY == 0 and step > 0:
            save_checkpoint(step)

finally:
    #this does not work as intended but is not really needed
    save_checkpoint(_last_step)

print("Training complete.")
wandb.finish()
ray.shutdown()
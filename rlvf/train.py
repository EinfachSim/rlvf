import os
import sys
import ray
import torch
import numpy as np
import wandb

# Local imports from your project
from rlvf.workers import EnvWorker
from rlvf.ppo import PPO
from rlvf.model import PolicyValueNetwork  # Adjust if your import path differs

# Configuration parameters
NUM_STEPS = 100
EPISODES_PER_STEP = 16
NUM_WORKERS = 2  # Matches available GPU resources (1 worker per GPU)
KL_WEIGHT = 0.1
LEARNING_RATE = 1e-4

def main():
    # 1. Initialize Ray cluster connection
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    print("Initializing Weights & Biases...")
    # 2. Initialize WandB logging
    wandb.init(
        project="rlvf-pvq-alignment",
        name="mistral-7b-lora-ppo",
        config={
            "model": "Mistral-7B-v0.3",
            "kl_weight": KL_WEIGHT,
            "learning_rate": LEARNING_RATE,
            "num_steps": NUM_STEPS,
            "episodes_per_step": EPISODES_PER_STEP,
            "num_workers": NUM_WORKERS,
        }
    )

    print(f"Instantiating {NUM_WORKERS} Ray EnvWorkers...")
    # 3. Create worker actors (1 GPU per worker)
    workers = [EnvWorker.remote() for _ in range(NUM_WORKERS)]

    print("Initializing PPO Controller...")
    # 4. Initialize PPO Policy & Value Network and Optimizer
    # Adjust state/action space dimensions according to your model spec
    policy_net = PolicyValueNetwork()  
    ppo = PPO(policy_net, lr=LEARNING_RATE)

    print("Starting training loop...")

    for step in range(NUM_STEPS):
        # Sample or generate random target profiles for this batch (e.g., 16 profiles)
        profiles = np.random.uniform(-1.0, 1.0, size=(EPISODES_PER_STEP, 10)).astype(np.float32)

        # 5. Evaluate PPO policy on profiles to get actions (z, A, B)
        with torch.no_grad():
            states = torch.tensor(profiles, dtype=torch.float32)
            # Sample actions and values from current policy
            actions, logp, values = ppo.pol.get_action_and_value(states)

        # Convert actions for Ray worker execution
        actions_np = actions.cpu().numpy()

        # Split batch evenly across workers
        profiles_split = np.array_split(profiles, NUM_WORKERS)
        actions_split = np.array_split(actions_np, NUM_WORKERS)

        # 6. Dispatch episodes across active workers
        futures = []
        for i, worker in enumerate(workers):
            f = worker.run_episodes_serial.remote(
                profiles=profiles_split[i].tolist(),
                actions=actions_split[i],
                kl_weight=KL_WEIGHT,
            )
            futures.append(f)

        # Gather worker results
        worker_results = ray.get(futures)
        episodes = [ep for sublist in worker_results for ep in sublist]

        # Extract environment metrics
        rewards = [ep["reward"] for ep in episodes]
        scores = [ep["score"] for ep in episodes]
        kls = [ep["kl"] for ep in episodes]

        # 7. Construct batch for PPO update
        batch = {
            "states": states,
            "actions": actions,
            "logp_old": logp,
            "values_old": values,
            "rewards": torch.tensor(rewards, dtype=torch.float32),
            "episodes": episodes,
        }

        # 8. Perform PPO update
        ppo_info = ppo.update(batch)

        # Compute summary metrics
        mean_reward = float(np.mean(rewards))
        mean_score = float(np.mean(scores))
        mean_kl = float(np.mean(kls))

        # 9. Log step metrics to WandB
        metrics = {
            "train/reward_mean": mean_reward,
            "train/reward_min": float(np.min(rewards)),
            "train/reward_max": float(np.max(rewards)),
            "train/score_mean": mean_score,
            "train/kl_mean": mean_kl,
            "step": step,
        }

        # Add PPO loss metrics if returned by ppo.update()
        if isinstance(ppo_info, dict):
            for k, v in ppo_info.items():
                if isinstance(v, torch.Tensor):
                    v = v.item()
                metrics[f"ppo/{k}"] = v

        wandb.log(metrics, step=step)

        # Console logging
        print(
            f"Step {step:03d}/{NUM_STEPS:03d} | "
            f"Reward: {mean_reward:8.4f} | "
            f"Score: {mean_score:8.4f} | "
            f"KL: {mean_kl:8.4f}"
        )

    # Clean close
    print("Training complete. Closing WandB run...")
    wandb.finish()


if __name__ == "__main__":
    main()
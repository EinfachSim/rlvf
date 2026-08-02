import torch

"""
CHANGES vs previous version
---------------------------
1. log-probs from the policy are now JOINT (see model.py), so `ratio` is the
   true importance ratio and the clip range actually bounds the policy step.
2. approx_kl: k1 estimator mean(logp_old - logp_new) on the joint log-probs
   (previously the k2 form 0.5*(dlogp)^2 was computed on per-dim-normalized
   log-probs, underestimating true KL by ~action_dim^2, so early stopping
   never fired: update_iters was 10 in all 52 logged steps, while the true
   per-update KL was ~100-240 nats).
3. Early stopping is checked BEFORE applying the epoch's gradient step, so the
   update that first exceeds the bound is not applied.
4. Added clip_frac diagnostic (fraction of samples where the clip is active).
"""


class PPO:
    def __init__(
            self,
            policy,
            num_iter=10,
            clip_ratio=0.2,
            lr=5e-5,
            vf_coef=0.5,
            ent_coef=0.0,
            target_kl=0.02,
            ):
        self.num_iterations = num_iter
        self.pol = policy
        self.clip_ratio = clip_ratio
        self.lr = lr
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.target_kl = target_kl
        self.optimizer = torch.optim.Adam(self.pol.parameters(), lr=lr)

    def update(self, batch):
        states, actions, rewards, logprobs = batch

        advantages = self._advantages(states, rewards)  # (batch_size,)

        # Advantage normalization
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        iters_done = 0
        stopped_early = False
        for i in range(self.num_iterations):
            logp_new, value, entropy = self._evaluate(states, actions)

            # KL early stop — k1 estimator on JOINT log-probs, checked BEFORE
            # stepping so we never apply an update that breaches the bound.
            with torch.no_grad():
                approx_kl = (logprobs - logp_new).mean().item()
            if i > 0 and approx_kl > 1.5 * self.target_kl:
                stopped_early = True
                break

            # Policy loss
            ratio = torch.exp(logp_new - logprobs)  # (batch_size,) TRUE ratio
            clipped = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
            loss_pi = -torch.min(ratio * advantages, clipped * advantages).mean()

            with torch.no_grad():
                clip_frac = (
                    (ratio - 1.0).abs() > self.clip_ratio
                ).float().mean().item()

            # Value loss
            loss_v = ((value - rewards) ** 2).mean()

            # Total loss
            loss = loss_pi + self.vf_coef * loss_v - self.ent_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.pol.parameters(), max_norm=0.5
            )
            self.optimizer.step()
            iters_done = i + 1

        with torch.no_grad():
            return {
                "loss_pi":      loss_pi.item(),
                "loss_v":       loss_v.item(),
                "entropy":      entropy.item(),
                # k1 KL of the *last evaluated* policy vs the behavior policy.
                "approx_kl":    approx_kl,
                "update_iters": iters_done,
                "stopped_early": float(stopped_early),
                # Log prob diagnostics — JOINT log-probs (no normalization)
                "logp_old_mean": logprobs.mean().item(),
                "logp_new_mean": logp_new.mean().item(),
                # Ratio diagnostics — true importance ratios
                "ratio_mean":   ratio.mean().item(),
                "ratio_max":    ratio.max().item(),
                "ratio_min":    ratio.min().item(),
                "clip_frac":    clip_frac,
                # Advantage diagnostics — mean ~0, std ~1 after normalization
                "adv_mean":     advantages.mean().item(),
                "adv_std":      advantages.std().item(),
                # Gradient norm before clipping — if near 0, gradients aren't flowing
                "grad_norm":    grad_norm.item(),
                # Value diagnostics
                "value_mean":   value.mean().item(),
                "value_std":    value.std().item(),
            }

    def _advantages(self, states, rewards):
        # Single-step (contextual bandit) setting: A(s, a) = r - V(s)
        with torch.no_grad():
            values = self.pol.get_value(states).squeeze(-1)
        return rewards - values

    def _evaluate(self, states, actions):
        # states: (batch_size, 19), actions: (batch_size, L*T, rank)
        logprob, mean_entropy = self.pol.get_action_and_logprob(
            states, action=actions, use_action=True
        )
        values = self.pol.get_value(states).squeeze(-1)
        return logprob, values, mean_entropy
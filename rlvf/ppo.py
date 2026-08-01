import torch

class PPO:
    def __init__(
            self,
            policy,
            num_iter=10,
            clip_ratio=0.2,
            lr=1e-2,
            vf_coef=0.5,
            ent_coef=0.0,
            target_kl=0.01
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

        advantages = self._batch_gae(states, rewards)  # (batch_size,)

        # Advantage normalization
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for i in range(self.num_iterations):
            logp_new, value, entropy = self._evaluate(states, actions)

            # Policy loss
            ratio = torch.exp(logp_new - logprobs)  # (batch_size,)
            clipped = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
            loss_pi = -torch.min(ratio * advantages, clipped * advantages).mean()

            # Value loss
            loss_v = ((value - rewards) ** 2).mean()

            # Total loss
            loss = loss_pi + self.vf_coef * loss_v - self.ent_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.pol.parameters(), max_norm=0.5)
            self.optimizer.step()

            # Early stopping
            with torch.no_grad():
                approx_kl = (0.5 * (logprobs - logp_new)**2).item()
            if approx_kl > 1.5 * self.target_kl:
                break

        with torch.no_grad():
            return {
                "loss_pi":      loss_pi.item(),
                "loss_v":       loss_v.item(),
                "entropy":      entropy.item(),
                "approx_kl":    approx_kl,
                "update_iters": i + 1,
                # Log prob diagnostics — should be small numbers close to 0 after normalization
                "logp_old_mean": logprobs.mean().item(),
                "logp_new_mean": logp_new.mean().item(),
                # Ratio diagnostics — should float around 1.0
                "ratio_mean":   ratio.mean().item(),
                "ratio_max":    ratio.max().item(),
                "ratio_min":    ratio.min().item(),
                # Advantage diagnostics — mean ~0, std ~1 after normalization
                "adv_mean":     advantages.mean().item(),
                "adv_std":      advantages.std().item(),
                # Gradient norm before clipping — if near 0, gradients aren't flowing
                "grad_norm":    grad_norm.item(),
                # Value diagnostics
                "value_mean":   value.mean().item(),
                "value_std":    value.std().item(),
            }

    def _batch_gae(self, states, rewards):
        with torch.no_grad():
            values = self.pol.get_value(states).squeeze(-1)
        return rewards - values

    def _evaluate(self, states, actions):
        # states: (batch_size, 19), actions: (batch_size, L*T, rank)
        logprob, mean_entropy = self.pol.get_action_and_logprob(states, action=actions, use_action=True)
        values = self.pol.get_value(states).squeeze(-1)
        return logprob, values, mean_entropy
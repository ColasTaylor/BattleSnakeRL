from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Categorical

    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

from snake_rl.agents.base import Agent

if _HAS_TORCH:

    class ActorCriticNet(nn.Module):
        """Simple shared backbone with separate policy/value heads."""

        def __init__(
            self, obs_shape: Sequence[int], num_actions: int, hidden_size: int = 256
        ) -> None:
            super().__init__()
            in_dim = int(np.prod(obs_shape))
            self.net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
            )
            self.policy_head = nn.Linear(hidden_size, num_actions)
            self.value_head = nn.Linear(hidden_size, 1)

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            z = self.net(x)
            logits = self.policy_head(z)
            value = self.value_head(z).squeeze(-1)
            return logits, value

else:

    class ActorCriticNet:  # type: ignore[no-redef]
        """Placeholder network when torch is unavailable."""

        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("PyTorch is required for PPOAgent.")


class PPOAgent(Agent):
    """PPO agent with GAE and minibatch updates."""

    def __init__(
        self,
        obs_shape: Sequence[int],
        num_actions: int,
        name: str = "ppo",
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_eps: float = 0.2,
        lr: float = 1e-4,
        rollout_length: int = 1024,
        num_epochs: int = 3,
        batch_size: int = 256,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
    ) -> None:
        if not _HAS_TORCH:
            raise ImportError("PyTorch is required for PPOAgent.")
        super().__init__(num_actions, name)
        self.obs_shape = tuple(obs_shape)
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.rollout_length = rollout_length
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = ActorCriticNet(self.obs_shape, num_actions, hidden_size=256).to(
            self.device
        )
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.reset_buffer()

    def reset_buffer(self) -> None:
        self.buffer: Dict[str, List] = {
            "obs": [],
            "actions": [],
            "logprobs": [],
            "rewards": [],
            "dones": [],
            "values": [],
        }

    def act(self, obs: np.ndarray, deterministic: bool = False) -> int:
        obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        logits, value = self.policy(obs_t)
        dist = Categorical(logits=logits)
        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            action = dist.sample()
        logprob = dist.log_prob(action)
        if not deterministic:
            # Only record transitions when training
            self.buffer["obs"].append(obs.copy())
            self.buffer["actions"].append(action.item())
            self.buffer["logprobs"].append(logprob.item())
            self.buffer["values"].append(value.detach().cpu().item())
        return int(action.item())

    def observe(self, transition: Dict[str, Any]) -> None:
        self.buffer["rewards"].append(transition["reward"])
        self.buffer["dones"].append(transition["done"])

    def _compute_gae(self) -> Tuple[np.ndarray, np.ndarray]:
        rewards = np.array(self.buffer["rewards"], dtype=np.float32)
        values = np.array(self.buffer["values"], dtype=np.float32)
        dones = np.array(self.buffer["dones"], dtype=np.float32)
        T = len(rewards)

        # Bootstrap from last state if non-terminal
        last_value = 0.0
        if T > 0 and dones[-1] == 0.0:
            last_value = values[-1]
        values = np.append(values, last_value)
        advantages = np.zeros_like(rewards)
        last_adv = 0.0
        for t in reversed(range(T)):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * values[t + 1] * mask - values[t]
            last_adv = delta + self.gamma * self.lam * mask * last_adv
            advantages[t] = last_adv
        returns = advantages + values[:-1]
        return advantages, returns

    def update(self) -> Dict[str, Any]:
        if len(self.buffer["rewards"]) < self.rollout_length:
            return {}
        obs = torch.from_numpy(np.array(self.buffer["obs"])).float().to(self.device)
        actions = torch.tensor(self.buffer["actions"]).long().to(self.device)
        old_logprobs = torch.tensor(self.buffer["logprobs"]).float().to(self.device)
        advantages, returns = self._compute_gae()
        advantages = torch.from_numpy(advantages).float().to(self.device)
        returns = torch.from_numpy(returns).float().to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        dataset_size = obs.shape[0]
        idxs = np.arange(dataset_size)
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_steps = 0
        for _ in range(self.num_epochs):
            np.random.shuffle(idxs)
            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                mb_idx = idxs[start:end]
                mb_obs = obs[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_logprobs = old_logprobs[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_returns = returns[mb_idx]
                logits, values = self.policy(mb_obs)
                dist = Categorical(logits=logits)
                new_logprobs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()
                ratio = torch.exp(new_logprobs - mb_old_logprobs)
                surr1 = ratio * mb_advantages
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
                    * mb_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = (mb_returns - values).pow(2).mean()
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy
                )
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimizer.step()
                total_policy_loss += policy_loss.item() * len(mb_idx)
                total_value_loss += value_loss.item() * len(mb_idx)
                total_entropy += entropy.item() * len(mb_idx)
                total_steps += len(mb_idx)
        stats = {
            "policy_loss": total_policy_loss / max(1, total_steps),
            "value_loss": total_value_loss / max(1, total_steps),
            "entropy": total_entropy / max(1, total_steps),
        }
        self.reset_buffer()
        return stats

    def save(self, path: str) -> None:
        """Save policy parameters to disk."""
        import torch

        state = {
            "policy": self.policy.state_dict(),
            "obs_shape": self.obs_shape,
            "num_actions": self.num_actions,
        }
        torch.save(state, path)

    def load(self, path: str) -> None:
        """Load policy parameters from disk."""
        import torch

        state = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(state["policy"])

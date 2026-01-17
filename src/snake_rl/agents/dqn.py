import random as pyrand
from collections import deque
from typing import Any, Deque, Dict, Sequence, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

from snake_rl.agents.base_agent import Agent

if _HAS_TORCH:

    class QNetwork(nn.Module):
        """Simple MLP Q-network.

        - Flattens the observation (whatever shape you give it).
        - Outputs Q(s, a) for each discrete action.
        """

        def __init__(
            self, obs_shape: Sequence[int], num_actions: int, hidden_size: int = 128
        ) -> None:
            super().__init__()
            in_dim = int(np.prod(obs_shape))

            self.net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, num_actions),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

else:

    class QNetwork:  # type: ignore[no-redef]
        """Placeholder network when torch is unavailable."""

        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("PyTorch is required for DQNAgent.")


class DQNAgent(Agent):
    """Vanilla DQN + Double DQN target.

    Notes:
    - It does NOT “know” anything about health or Battlesnake rules.
      It just sees whatever observation array you give it.
    - If you add health or other features, just make sure the env's `obs`
      includes them (e.g., extra channels or stacked features); this agent
      will automatically adapt because it only depends on `obs_shape`.
    """

    def __init__(
        self,
        obs_shape: Sequence[int],
        num_actions: int,
        name: str = "dqn",
        gamma: float = 0.99,
        lr: float = 1e-3,
        buffer_size: int = 100_000,
        batch_size: int = 128,
        min_buffer: int = 5_000,
        # Exploration schedule
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.10,
        epsilon_decay_steps: int = 200_000,
        # Target network sync frequency (in env steps)
        target_update_freq: int = 5_000,
    ) -> None:
        if not _HAS_TORCH:
            raise ImportError("PyTorch is required for DQNAgent.")

        super().__init__(num_actions, name)

        self.obs_shape = tuple(obs_shape)
        self.gamma = gamma
        self.batch_size = batch_size
        self.min_buffer = min_buffer
        self.target_update_freq = target_update_freq

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Online and target networks
        self.q_net = QNetwork(self.obs_shape, num_actions).to(self.device)
        self.target_net = QNetwork(self.obs_shape, num_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        # Replay buffer: (obs, action, reward, next_obs, done)
        self.replay: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(
            maxlen=buffer_size
        )

        # Epsilon-greedy exploration
        self.total_steps = 0
        self.eps_start = epsilon_start
        self.eps_end = epsilon_end
        self.eps_decay_steps = epsilon_decay_steps

    # ----------------- Action selection -----------------

    def _epsilon(self) -> float:
        """Linear epsilon decay from eps_start → eps_end over epsilon_decay_steps.
        After that, stays at eps_end.
        """
        frac = min(1.0, self.total_steps / float(self.eps_decay_steps))
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def act(self, obs: np.ndarray, deterministic: bool = False) -> int:
        """Epsilon-greedy policy.

        - deterministic=True: no exploration (eps=0), do NOT increment total_steps.
          Used for evaluation.
        - deterministic=False: normal training mode, eps decays over time and
          total_steps is incremented at each call.
        """
        if deterministic:
            eps = 0.0
        else:
            eps = self._epsilon()
            self.total_steps += 1

        # Exploration
        if np.random.rand() < eps:
            return np.random.randint(self.num_actions)

        # Exploitation
        obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(obs_t)
        action = int(torch.argmax(q_values, dim=-1).item())
        return action

    # ----------------- Replay + learning -----------------

    def observe(self, transition: Dict[str, Any]) -> None:
        """Store a single transition in the replay buffer.

        transition keys:
            "obs": np.ndarray
            "action": int
            "reward": float
            "next_obs": np.ndarray
            "done": bool
        """
        self.replay.append(
            (
                transition["obs"],
                transition["action"],
                transition["reward"],
                transition["next_obs"],
                transition["done"],
            )
        )

    def update(self) -> Dict[str, Any]:
        """One gradient step of DQN (with Double DQN target).
        Returns a dict with "dqn_loss" if an update was performed, {} otherwise.
        """
        if len(self.replay) < self.min_buffer:
            return {}

        # Sample a random minibatch
        batch = pyrand.sample(self.replay, self.batch_size)
        obs_batch, act_batch, rew_batch, next_obs_batch, done_batch = zip(*batch)

        obs_t = torch.from_numpy(np.stack(obs_batch)).float().to(self.device)
        acts_t = torch.tensor(act_batch).long().to(self.device)
        rews_t = torch.tensor(rew_batch).float().to(self.device)
        next_obs_t = torch.from_numpy(np.stack(next_obs_batch)).float().to(self.device)
        done_t = torch.tensor(done_batch).float().to(self.device)

        # Q(s, a) for the taken actions
        q_values = self.q_net(obs_t)  # [B, num_actions]
        q_sa = q_values.gather(1, acts_t.unsqueeze(1)).squeeze(1)  # [B]

        # ----- Double DQN target -----
        with torch.no_grad():
            # Online net selects best action in next state
            next_q_online = self.q_net(next_obs_t)  # [B, num_actions]
            next_actions = next_q_online.argmax(dim=1)  # [B]

            # Target net evaluates those actions
            next_q_target = self.target_net(next_obs_t)  # [B, num_actions]
            next_q_target_sa = next_q_target.gather(
                1, next_actions.unsqueeze(1)
            ).squeeze(1)  # [B]

            target = rews_t + self.gamma * (1.0 - done_t) * next_q_target_sa

        # ----- Loss + optimization -----
        loss = (q_sa - target).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=5.0)
        self.optimizer.step()

        # Periodically sync target network
        if self.total_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return {"dqn_loss": loss.item()}

    # ----------------- Save / load -----------------

    def save(self, path: str) -> None:
        """Save Q-network parameters to disk."""
        import torch

        state = {
            "q_net": self.q_net.state_dict(),
            "obs_shape": self.obs_shape,
            "num_actions": self.num_actions,
        }
        torch.save(state, path)

    def load(self, path: str) -> None:
        """Load Q-network parameters from disk."""
        import torch

        state = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(state["q_net"])
        self.target_net.load_state_dict(self.q_net.state_dict())

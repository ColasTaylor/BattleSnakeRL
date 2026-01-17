# hmm_opponent_model.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from snake_rl.envs.snake_env import SnakeGridEnv


# ============================================================
# 1. Generic discrete HMM
# ============================================================


@dataclass
class HMMConfig:
    """Configuration for a discrete HMM."""

    num_states: int
    num_obs_symbols: int
    stay_prob: float = 0.95
    seed: Optional[int] = None


class DiscreteHMM:
    """Tiny discrete HMM with:

    - K hidden states (styles)
    - M discrete observations (symbols)
    - Online filtering belief update.

    We use it only in filtering mode:
        b_t ∝ (b_{t-1} @ A) ⊙ B[:, o_t]
    """

    def __init__(self, cfg: HMMConfig) -> None:
        self.K = cfg.num_states
        self.M = cfg.num_obs_symbols
        self.rng = np.random.RandomState(cfg.seed)

        # Transition matrix A (K x K), diagonally dominant:
        stay = cfg.stay_prob
        if self.K > 1:
            off = (1.0 - stay) / (self.K - 1)
        else:
            off = 0.0
        self.A = np.full((self.K, self.K), off, dtype=np.float32)
        np.fill_diagonal(self.A, stay)

        # Emission matrix B (K x M), initialize uniform
        self.B = np.full((self.K, self.M), 1.0 / self.M, dtype=np.float32)

        # Initial state distribution π (uniform)
        self.pi = np.full(self.K, 1.0 / self.K, dtype=np.float32)

        # Current belief b_t
        self.belief = self.pi.copy()

    # ----- public API -----

    def reset(self) -> None:
        """Reset belief to the prior."""
        self.belief = self.pi.copy()

    def set_emissions(self, B: np.ndarray) -> None:
        """Set custom emission matrix B (K x M)."""
        assert B.shape == (self.K, self.M)
        self.B = B.astype(np.float32)

    def update(self, obs_symbol: int) -> None:
        """One filtering update for observation o_t.

        obs_symbol must be in [0, M-1].
        """
        assert 0 <= obs_symbol < self.M, "obs_symbol out of range"

        # Predict: prior over z_t
        pred = self.belief @ self.A  # shape (K,)

        # Update: multiply by emission probabilities for o_t
        pred *= self.B[:, obs_symbol]

        s = pred.sum()
        if s <= 0.0:
            # Degenerate: revert to uniform
            self.belief = np.full(self.K, 1.0 / self.K, dtype=np.float32)
        else:
            self.belief = pred / s

    def get_belief(self) -> np.ndarray:
        """Return current belief b_t (shape [K])."""
        return self.belief.copy()

    def most_likely_state(self) -> int:
        """Return argmax over current belief."""
        return int(np.argmax(self.belief))


# ============================================================
# 2. HMMOpponentModel tied to SnakeGridEnv
# ============================================================


class HMMOpponentModel:
    """Opponent-style HMM for SnakeGridEnv (2-snake setting).

    Hidden states (styles), by default:
        0: "hungry"  - moves toward food a lot
        1: "scared"  - moves away from us / danger
        2: "loopy"   - turns often (not keeping heading)
        3: "right"   - prefers moving right / straight

    Observations are discrete symbols based on FOUR binary features:

      - toward_food (tf):   1 if opponent moved closer to nearest food
      - away_enemy (ae):    1 if opponent's distance to our head increased
      - keep_heading (kh):  1 if direction_t == direction_{t-1}
      - moved_right (mr):   1 if x_new > x_old (for non-wrapping boards)

    We encode these 4 bits into a symbol:
        symbol = tf + 2 * ae + 4 * kh + 8 * mr    in {0, ..., 15}

    Usage pattern in training:
      env = SnakeGridEnv(num_snakes=2, ...)
      hmm_model = HMMOpponentModel()

      obs_n = env.reset()
      hmm_model.reset(env, learning_idx=0, opponent_idx=1)

      while not done:
          # RL picks actions ...
          obs_n, rew_n, done, info = env.step(actions)
          hmm_model.update_from_env(env)   # updates belief

          belief = hmm_model.get_belief()  # shape [4]
          my_obs = obs_n[0].flatten()
          aug_obs = np.concatenate([my_obs, belief], axis=0)
    """

    def __init__(
        self,
        style_names: Optional[Sequence[str]] = None,
        stay_prob: float = 0.95,
        seed: Optional[int] = 42,
    ) -> None:
        if style_names is None:
            style_names = ["hungry", "scared", "loopy", "right"]
        assert len(style_names) >= 1
        self.style_names = list(style_names)
        self.num_states = len(style_names)
        self.num_obs_symbols = 16  # 4 bits -> 16 symbols

        cfg = HMMConfig(
            num_states=self.num_states,
            num_obs_symbols=self.num_obs_symbols,
            stay_prob=stay_prob,
            seed=seed,
        )
        self.hmm = DiscreteHMM(cfg)
        self._init_default_emissions()

        # Indices of learning snake / opponent in env.snakes
        self.learning_idx: int = 0
        self.opponent_idx: int = 1

        # Previous positions / directions for feature computation
        self._prev_opp_head: Optional[Tuple[int, int]] = None
        self._prev_opp_dir: Optional[int] = None
        self._prev_me_head: Optional[Tuple[int, int]] = None

    # ----- initialization -----

    def _init_default_emissions(self) -> None:
        """Build a simple hand-crafted emission matrix B (K x 16)
        using the feature bits described in the class docstring.

        Heuristics:
          - hungry: high prob when toward_food = 1
          - scared: high prob when away_enemy = 1
          - loopy: high prob when keep_heading = 0 (lots of turns)
          - right: high prob when moved_right = 1 and keep_heading = 1
        """
        K = self.num_states
        M = self.num_obs_symbols
        B = np.zeros((K, M), dtype=np.float32)

        for state_idx, name in enumerate(self.style_names):
            w = np.ones(M, dtype=np.float32)

            for s in range(M):
                tf = s & 1  # toward_food
                ae = (s >> 1) & 1  # away_enemy
                kh = (s >> 2) & 1  # keep_heading
                mr = (s >> 3) & 1  # moved_right

                # Base weight
                weight = 1.0

                if name.lower().startswith("hungry"):
                    # Strong preference for moving toward food
                    if tf == 1:
                        weight *= 4.0
                    if ae == 1:
                        weight *= 1.1  # slightly okay to move away from us

                elif name.lower().startswith("scared"):
                    # Strong preference for moving away from us (enemy head)
                    if ae == 1:
                        weight *= 4.0
                    # mild preference for not rushing food directly
                    if tf == 1:
                        weight *= 1.1

                elif name.lower().startswith("loopy"):
                    # Likes to turn (not keep heading)
                    if kh == 0:
                        weight *= 4.0
                    else:
                        weight *= 0.5

                elif name.lower().startswith("right"):
                    # Likes moving right and often straight
                    if mr == 1:
                        weight *= 4.0
                    else:
                        weight *= 0.4
                    if kh == 1:
                        weight *= 1.5

                else:
                    # Unknown style name -> keep uniform
                    pass

                w[s] = weight

            # Normalize row
            if w.sum() <= 0:
                B[state_idx, :] = 1.0 / M
            else:
                B[state_idx, :] = w / w.sum()

        self.hmm.set_emissions(B)

    # ----- public API -----

    def reset(
        self, env: SnakeGridEnv, learning_idx: int = 0, opponent_idx: int = 1
    ) -> None:
        """Reset the HMM belief and snapshot positions from the given env.

        Call this right after env.reset().
        """
        assert env.num_snakes == 2, "HMMOpponentModel assumes 2 snakes"
        assert 0 <= learning_idx < env.num_snakes
        assert 0 <= opponent_idx < env.num_snakes
        assert learning_idx != opponent_idx

        self.learning_idx = learning_idx
        self.opponent_idx = opponent_idx

        self.hmm.reset()
        self._snapshot_from_env(env)

    def update_from_env(self, env: SnakeGridEnv) -> None:
        """Update the HMM based on the *change* in opponent behaviour between
        previous snapshot and current env state.

        Call this after each env.step(...).
        """
        if self._prev_opp_head is None:
            # No previous info yet; just snapshot and return.
            self._snapshot_from_env(env)
            return

        # Current states
        opp = env.snakes[self.opponent_idx]
        me = env.snakes[self.learning_idx]
        if not opp.alive:
            # If opponent died, we could emit a special symbol; for now, stop updating
            self._snapshot_from_env(env)
            return

        new_opp_head = opp.body[0]
        new_opp_dir = opp.direction
        new_me_head = me.body[0]

        symbol = self._encode_observation_symbol(
            env,
            prev_opp_head=self._prev_opp_head,
            prev_opp_dir=self._prev_opp_dir,
            prev_me_head=self._prev_me_head,
            new_opp_head=new_opp_head,
            new_opp_dir=new_opp_dir,
            new_me_head=new_me_head,
        )

        self.hmm.update(symbol)

        # Move snapshot forward for next step
        self._snapshot_from_env(env)

    def get_belief(self) -> np.ndarray:
        """Return current belief over styles (shape [num_states])."""
        return self.hmm.get_belief()

    def get_style_names(self) -> List[str]:
        return list(self.style_names)

    def most_likely_style(self) -> str:
        idx = self.hmm.most_likely_state()
        return self.style_names[idx]

    # ----- internal helpers -----

    def _snapshot_from_env(self, env: SnakeGridEnv) -> None:
        opp = env.snakes[self.opponent_idx]
        me = env.snakes[self.learning_idx]

        self._prev_opp_head = opp.body[0]
        self._prev_opp_dir = opp.direction
        self._prev_me_head = me.body[0]

    def _nearest_food_distance(self, env: SnakeGridEnv, head: Tuple[int, int]) -> int:
        if not env.food_positions:
            return 0
        hx, hy = head
        return min(abs(hx - fx) + abs(hy - fy) for (fx, fy) in env.food_positions)

    def _encode_observation_symbol(
        self,
        env: SnakeGridEnv,
        prev_opp_head: Tuple[int, int],
        prev_opp_dir: int,
        prev_me_head: Tuple[int, int],
        new_opp_head: Tuple[int, int],
        new_opp_dir: int,
        new_me_head: Tuple[int, int],
    ) -> int:
        """Compute the 4 binary features and pack them into an int [0, 15].

        Bits (LSB -> MSB):
          bit 0: toward_food (tf)
          bit 1: away_enemy (ae)
          bit 2: keep_heading (kh)
          bit 3: moved_right (mr)
        """

        # 1) toward_food?
        old_food_dist = self._nearest_food_distance(env, prev_opp_head)
        new_food_dist = self._nearest_food_distance(env, new_opp_head)
        tf = 1 if new_food_dist < old_food_dist else 0

        # 2) away_enemy? (enemy = us)
        old_enemy_dist = abs(prev_opp_head[0] - prev_me_head[0]) + abs(
            prev_opp_head[1] - prev_me_head[1]
        )
        new_enemy_dist = abs(new_opp_head[0] - new_me_head[0]) + abs(
            new_opp_head[1] - new_me_head[1]
        )
        ae = 1 if new_enemy_dist > old_enemy_dist else 0

        # 3) keep_heading?
        kh = 1 if new_opp_dir == prev_opp_dir else 0

        # 4) moved_right? (simple x_new > x_old; works best for wrap=False)
        x_old, _ = prev_opp_head
        x_new, _ = new_opp_head
        mr = 1 if x_new > x_old else 0

        symbol = tf + 2 * ae + 4 * kh + 8 * mr
        return symbol

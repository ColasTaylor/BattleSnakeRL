from typing import Optional

import numpy as np

from snake_rl.agents.base import Agent
from snake_rl.envs.snake_grid_env import DIRS


class RandomAgent(Agent):
    """Uniform-random action baseline."""

    def __init__(self, num_actions: int, name: str = "random") -> None:
        super().__init__(num_actions, name)

    def act(self, obs: np.ndarray, deterministic: bool = False) -> int:
        return np.random.randint(self.num_actions)


class GreedyFoodAgent(Agent):
    """Simple baseline: step toward the nearest food while avoiding collisions."""

    def __init__(self, num_actions: int, name: str = "greedy_food") -> None:
        super().__init__(num_actions, name)

    def act(self, obs: np.ndarray, deterministic: bool = False) -> int:
        # obs encoding (from SnakeGridEnv._build_obs_for_snake):
        #   0 = empty
        #   1 = food
        #   2 = own head
        #   3 = own body
        #   4 = other snake(s)
        head_positions = np.argwhere(obs == 2)
        if len(head_positions) == 0:
            return np.random.randint(self.num_actions)
        hy, hx = head_positions[0]

        food_positions = np.argwhere(obs == 1)
        if len(food_positions) == 0:
            return np.random.randint(self.num_actions)
        fy, fx = food_positions[0]

        best_action = None
        best_dist = float("inf")
        h, w = obs.shape

        for a, (dx, dy) in DIRS.items():
            if a >= self.num_actions:
                continue
            nx = hx + dx
            ny = hy + dy
            if not (0 <= nx < w and 0 <= ny < h):
                continue
            cell = obs[ny, nx]
            if cell in (3, 4):  # own body or other snake = dangerous
                continue
            dist = abs(fx - nx) + abs(fy - ny)
            if dist < best_dist:
                best_dist = dist
                best_action = a

        if best_action is None:
            return np.random.randint(self.num_actions)
        return best_action


class SafeSpaceHeuristicAgent(Agent):
    """
    Heuristic agent usable for BOTH:
      - num_snakes = 1   (no '4' cells)
      - num_snakes = 2   (4 = other snake tiles)

    For each action:
      - Reject moves that hit walls or bodies.
      - Score remaining moves by:
          3.0 * reachable-space   (don't trap yourself)
        + 2.0 * food-term         (move towards food)
        + 0.7 * wall-term         (avoid hugging wall too much)
        + 0.3 * center-term       (stay somewhat central)
        - 0.5 * opponent-penalty  (stay a bit away from opponent if present)
    """

    def __init__(self, num_actions: int, name: str = "heuristic_safe") -> None:
        super().__init__(num_actions, name)

    def act(self, obs: np.ndarray, deterministic: bool = False) -> int:
        h, w = obs.shape

        head_positions = np.argwhere(obs == 2)
        if len(head_positions) == 0:
            return np.random.randint(self.num_actions)
        hy, hx = head_positions[0]

        food_positions = np.argwhere(obs == 1)
        has_food = len(food_positions) > 0
        if has_food:
            fy, fx = food_positions[0]
            max_food_dist = h + w

        opp_positions = np.argwhere(obs == 4)
        has_opp = len(opp_positions) > 0

        best_action = None
        best_score = -1e9

        for a, (dx, dy) in DIRS.items():
            if a >= self.num_actions:
                continue

            nx = hx + dx
            ny = hy + dy

            if not (0 <= nx < w and 0 <= ny < h):
                continue

            cell = obs[ny, nx]
            if cell in (3, 4):
                continue

            space = self._reachable_space(obs, nx, ny)
            space_score = space / float(h * w)

            food_score = 0.0
            if has_food:
                dist_food = abs(fx - nx) + abs(fy - ny)
                food_score = 1.0 - (dist_food / max_food_dist)

            wall_dist = self._distance_to_wall(nx, ny, dx, dy, w, h)
            max_wall_dist = max(h, w)
            wall_score = wall_dist / float(max_wall_dist)

            cx, cy = w // 2, h // 2
            dist_center = abs(cx - nx) + abs(cy - ny)
            max_center_dist = h + w
            center_score = 1.0 - (dist_center / max_center_dist)

            opp_penalty = 0.0
            if has_opp:
                dists = np.abs(opp_positions[:, 1] - nx) + np.abs(
                    opp_positions[:, 0] - ny
                )
                dmin_opp = int(dists.min())
                opp_penalty = max(0, 2 - dmin_opp) / 2.0

            score = (
                3.0 * space_score  # prioritize not trapping yourself
                + 1.5 * food_score  # still moves toward food, but less aggressively
                + 0.7 * wall_score  # prefer moves that keep distance from walls
                + 0.3 * center_score  # gently prefer central positions
                - 0.5 * opp_penalty  # stronger repulsion from opponent if present
            )

            if score > best_score:
                best_score = score
                best_action = a

        if best_action is None:
            legal_actions = [a for a in range(self.num_actions)]
            return int(np.random.choice(legal_actions))

        return int(best_action)

    def _reachable_space(self, obs: np.ndarray, start_x: int, start_y: int) -> int:
        """Flood-fill from (start_x, start_y) treating bodies (3, 4) as walls."""
        h, w = obs.shape
        visited = np.zeros_like(obs, dtype=bool)
        stack = [(start_x, start_y)]
        blocked_values = {3, 4}

        count = 0
        while stack:
            x, y = stack.pop()
            if not (0 <= x < w and 0 <= y < h):
                continue
            if visited[y, x]:
                continue

            if obs[y, x] in blocked_values and not (x == start_x and y == start_y):
                continue

            visited[y, x] = True
            count += 1

            for dx, dy in DIRS.values():
                nx = x + dx
                ny = y + dy
                if 0 <= nx < w and 0 <= ny < h and not visited[ny, nx]:
                    stack.append((nx, ny))
        return count

    def _distance_to_wall(
        self, x: int, y: int, dx: int, dy: int, w: int, h: int
    ) -> int:
        """Steps forward in (dx, dy) until boundary; returns reachable distance."""
        dist = 0
        cx, cy = x, y
        while True:
            cx += dx
            cy += dy
            if not (0 <= cx < w and 0 <= cy < h):
                break
            dist += 1
        return dist


class LoopyBotAgent(Agent):
    """
    'Loopy' style bot:
      - Tries to keep turning in a consistent direction (right turn relative to last move).
      - Falls back to any safe move if its preferred turn would crash.
      - If totally stuck, picks a random action.
    """

    def __init__(self, num_actions: int, name: str = "loopy_bot") -> None:
        super().__init__(num_actions, name)
        self.last_action: Optional[int] = None

    def act(self, obs: np.ndarray, deterministic: bool = False) -> int:
        h, w = obs.shape

        # Find own head
        head_positions = np.argwhere(obs == 2)
        if len(head_positions) == 0:
            return np.random.randint(self.num_actions)
        hy, hx = head_positions[0]

        def is_safe(a: int) -> bool:
            dx, dy = DIRS[a]
            nx, ny = hx + dx, hy + dy
            if not (0 <= nx < w and 0 <= ny < h):
                return False
            cell = obs[ny, nx]
            # treat any body or other snake as unsafe; food is allowed
            if cell in (2, 3, 4):
                return False
            return True

        # Decide candidate order
        if self.last_action is None:
            # First move: arbitrary preference order
            candidate_order = [3, 0, 1, 2]  # right, up, down, left
        else:
            # Turn right relative to last direction, then straight, then left, then reverse
            right_turn = (self.last_action + 1) % 4
            straight = self.last_action
            left_turn = (self.last_action + 3) % 4
            reverse = (self.last_action + 2) % 4
            candidate_order = [right_turn, straight, left_turn, reverse]

        for a in candidate_order:
            if a < self.num_actions and is_safe(a):
                self.last_action = a
                return a

        # Fallback: any safe move
        safe_actions = [a for a in range(self.num_actions) if is_safe(a)]
        if safe_actions:
            a = int(np.random.choice(safe_actions))
            self.last_action = a
            return a

        # Completely stuck
        a = np.random.randint(self.num_actions)
        self.last_action = a
        return a


class RightBotAgent(Agent):
    """
    'Right' style bot:
      - Always tries to move RIGHT if safe.
      - If not safe, falls back to a fixed order of other directions.
      - Very simple, highly biased behaviour.
    """

    def __init__(self, num_actions: int, name: str = "right_bot") -> None:
        super().__init__(num_actions, name)

    def act(self, obs: np.ndarray, deterministic: bool = False) -> int:
        h, w = obs.shape

        head_positions = np.argwhere(obs == 2)
        if len(head_positions) == 0:
            return np.random.randint(self.num_actions)
        hy, hx = head_positions[0]

        def is_safe(a: int) -> bool:
            dx, dy = DIRS[a]
            nx, ny = hx + dx, hy + dy
            if not (0 <= nx < w and 0 <= ny < h):
                return False
            cell = obs[ny, nx]
            if cell in (2, 3, 4):  # own head/body or other snake
                return False
            return True

        # Prefer RIGHT (action 3), then some arbitrary fallback order
        preferred = [3, 0, 1, 2]  # right -> up -> down -> left
        for a in preferred:
            if a < self.num_actions and is_safe(a):
                return a

        # Any safe action
        safe_actions = [a for a in range(self.num_actions) if is_safe(a)]
        if safe_actions:
            return int(np.random.choice(safe_actions))

        # Stuck
        return np.random.randint(self.num_actions)

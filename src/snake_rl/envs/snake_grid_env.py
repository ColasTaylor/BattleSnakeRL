import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# Optional turtle rendering
try:
    import turtle
    from turtle import Screen, Turtle

    _HAS_TURTLE = True
except Exception:
    _HAS_TURTLE = False

# Directions: up, down, left, right
DIRS = {
    0: (0, 1),  # up    (note: y increases downward in our grid)
    1: (0, -1),  # down
    2: (-1, 0),  # left
    3: (1, 0),  # right
}

OPPOSITE = {
    0: 1,
    1: 0,
    2: 3,
    3: 2,
}


@dataclass
class SnakeState:
    """Container for per-snake state within the grid environment."""

    body: List[Tuple[int, int]]  # head is body[0]
    direction: int  # one of {0,1,2,3}
    alive: bool = True
    score: int = 0
    health: int = 100  # Battlesnake-style health


class SnakeGridEnv:
    """Battlesnake-style Snake environment.

    Grid encoding (internal, for rendering only):
      0 = empty
      1 = food
      10 + i = snake i body
      20 + i = snake i head

    Observations to agents are a grid of int8 with:
      0 = empty
      1 = food
      2 = own head
      3 = own body
      4 = opponent head / body
    """

    def __init__(
        self,
        grid_size: int = 15,
        num_snakes: int = 1,
        max_steps: int = 500,
        wrap: bool = False,
        render_mode: str = "none",  # "none" or "turtle"
        food_spawn_chance: int = 5,  # % chance per turn to spawn extra food
        minimum_food: int = 1,  # minimum food on board
        max_health: int = 100,  # Battlesnake-style max health
    ):
        assert num_snakes in (1, 2), "Only 1 or 2 snakes are supported in this scaffold"
        assert render_mode in ("none", "turtle")

        self.grid_size = grid_size
        self.num_snakes = num_snakes
        self.max_steps = max_steps
        self.wrap = wrap
        self.render_mode = render_mode

        # Grid encoding (internal):
        # 0 = empty
        # 1 = food
        # 10 + i = snake i body
        # 20 + i = snake i head
        self.grid = np.zeros((grid_size, grid_size), dtype=np.int8)
        self.snakes: List[SnakeState] = []

        # Battlesnake-style food / health
        self.food_positions: List[Tuple[int, int]] = []
        self.step_count: int = 0
        self.max_health = max_health
        self.food_spawn_chance = food_spawn_chance
        self.minimum_food = minimum_food

        # Turtle-related
        self._screen: Optional[Screen] = None
        self._drawer: Optional[Turtle] = None
        self._cell_px = 20

    # ------------- Public API -------------

    def reset(self) -> List[np.ndarray]:
        """Reset the environment and return initial observations for each snake."""
        self.grid.fill(0)
        self.snakes = []
        self.food_positions = []
        self.step_count = 0

        # Spawn snakes
        if self.num_snakes == 1:
            init_positions = [self._center_pos()]
        else:
            g = self.grid_size
            init_positions = [(g // 4, g // 2), (3 * g // 4, g // 2)]

        for i, (x, y) in enumerate(init_positions):
            direction = 3 if i == 0 else 2  # first goes right, second left
            dx, dy = DIRS[direction]
            # Start length 3: head plus two trailing segments
            body = [(x, y)]
            for k in range(1, 3):
                bx = x - dx * k
                by = y - dy * k
                if self.wrap:
                    bx %= self.grid_size
                    by %= self.grid_size
                body.append((bx, by))
            self.snakes.append(
                SnakeState(
                    body=body,
                    direction=direction,
                    alive=True,
                    score=0,
                    health=self.max_health,
                )
            )

        # Place initial food(s)
        self._ensure_minimum_food()

        # Build grid with initial snakes/food so render shows full bodies
        self._write_grid_from_state()

        if self.render_mode == "turtle":
            self._init_turtle()

        return self._build_all_obs()

    def step(
        self, actions: Union[int, Sequence[int]]
    ) -> Tuple[List[np.ndarray], List[float], bool, Dict[str, Any]]:
        """Advance the environment by one step.

        - All alive snakes choose a direction (we disallow 180° turns).
        - All heads move simultaneously.
        - Collisions are resolved (walls, self/body, head-to-head).
        - Survivors lose 1 health; snakes that ate food reset to full health and grow by 1.
        - Food is consumed and respawned according to minimum_food and food_spawn_chance.

        actions:
            - if num_snakes == 1: int ∈ {0,1,2,3} or [int]
            - if num_snakes == 2: [a0, a1]
        """
        if isinstance(actions, int):
            actions = [actions]
        assert len(actions) == self.num_snakes

        self.step_count += 1
        rewards = [0.0 for _ in range(self.num_snakes)]
        done = False
        info: Dict[str, Any] = {}

        # Small step reward for staying alive (survival incentive)
        for i in range(self.num_snakes):
            if self.snakes[i].alive:
                rewards[i] += 0.001

        # 1) Update directions based on actions (disallow 180° turns)
        for i, a in enumerate(actions):
            snake = self.snakes[i]
            if not snake.alive:
                continue
            a = int(a)
            if a not in DIRS:
                continue
            if OPPOSITE[a] == snake.direction:
                # ignore 180° turn
                continue
            snake.direction = a

        # 2) Compute new head positions
        new_heads: List[Tuple[int, int]] = []
        for i, snake in enumerate(self.snakes):
            if not snake.alive:
                new_heads.append(snake.body[0])  # dummy
                continue
            dx, dy = DIRS[snake.direction]
            hx, hy = snake.body[0]
            nx, ny = hx + dx, hy + dy
            if self.wrap:
                nx %= self.grid_size
                ny %= self.grid_size
            new_heads.append((nx, ny))

        # Track eliminations this turn
        eliminated = [False for _ in range(self.num_snakes)]

        # 3) Wall collisions (if not wrapping)
        for i, snake in enumerate(self.snakes):
            if not snake.alive or self.wrap:
                continue
            nx, ny = new_heads[i]
            if not self._in_bounds(nx, ny):
                eliminated[i] = True
                rewards[i] -= 1.0  # death penalty

        # 4) Head-to-head collisions (Battlesnake-style: longest survives)
        # Map head positions to indices of snakes that move there
        head_counts: Dict[Tuple[int, int], List[int]] = {}
        for i, snake in enumerate(self.snakes):
            if not snake.alive or eliminated[i]:
                continue
            pos = new_heads[i]
            if not self._in_bounds(pos[0], pos[1]) and not self.wrap:
                # already handled wall collisions above
                continue
            head_counts.setdefault(pos, []).append(i)

        for pos, idxs in head_counts.items():
            if len(idxs) <= 1:
                continue
            lengths = [len(self.snakes[i].body) for i in idxs]
            max_len = max(lengths)
            winners = [idxs[k] for k, L in enumerate(lengths) if L == max_len]
            if len(winners) == 1:
                winner = winners[0]
                for i in idxs:
                    if i == winner:
                        continue
                    if not eliminated[i]:
                        eliminated[i] = True
                        rewards[i] -= 1.0
            else:
                # tie: everyone at this head position dies
                for i in idxs:
                    if not eliminated[i]:
                        eliminated[i] = True
                        rewards[i] -= 1.0

        # 5) Determine which snakes would eat food (for growth & tail safety)
        will_grow = [False for _ in range(self.num_snakes)]
        food_set = set(self.food_positions)
        for i, snake in enumerate(self.snakes):
            if not snake.alive or eliminated[i]:
                continue
            if new_heads[i] in food_set:
                will_grow[i] = True

        # Precompute body segments (before movement)
        bodies_without_tail: List[set] = []
        tails: List[Optional[Tuple[int, int]]] = []
        for snake in self.snakes:
            if len(snake.body) <= 1:
                bodies_without_tail.append(set())
                tails.append(snake.body[0] if snake.body else None)
            else:
                bodies_without_tail.append(set(snake.body[:-1]))
                tails.append(snake.body[-1])

        # 6) Body collisions (self and others, with safe-tail behaviour)
        for i, snake in enumerate(self.snakes):
            if not snake.alive or eliminated[i]:
                continue
            nx, ny = new_heads[i]

            # Out-of-bounds already handled if wrap=False; but guard anyway
            if not self.wrap and not self._in_bounds(nx, ny):
                continue

            pos = (nx, ny)

            # Self-collision with own body (excluding tail)
            if pos in bodies_without_tail[i]:
                eliminated[i] = True
                rewards[i] -= 1.0
                continue

            # Moving into own tail:
            # - Safe if we are NOT growing (tail will move away)
            # - Unsafe if we ARE growing (tail stays)
            if tails[i] is not None and pos == tails[i] and will_grow[i]:
                eliminated[i] = True
                rewards[i] -= 1.0
                continue

            # Collisions with other snakes
            for j, other in enumerate(self.snakes):
                if j == i or not other.alive:
                    continue

                # Into other snake's body (excluding tail) is always deadly
                if pos in bodies_without_tail[j]:
                    eliminated[i] = True
                    rewards[i] -= 1.0
                    break

                # Into other snake's tail:
                # - Safe if that snake is NOT growing (its tail moves)
                # - Unsafe if that snake IS growing (tail stays)
                if tails[j] is not None and pos == tails[j] and will_grow[j]:
                    eliminated[i] = True
                    rewards[i] -= 1.0
                    break

        # 7) Move snakes, apply growth & health updates
        new_bodies: List[List[Tuple[int, int]]] = []
        for i, snake in enumerate(self.snakes):
            if not snake.alive:
                new_bodies.append(snake.body)
                continue

            if eliminated[i]:
                # Snake dies this turn; body stays where it was
                snake.alive = False
                new_bodies.append(snake.body)
                continue

            # Survivor: move head and shift body
            nx, ny = new_heads[i]
            old_body = snake.body
            new_body = [(nx, ny)] + old_body[:]  # prepend new head

            if not will_grow[i]:
                # Tail moves forward (no growth)
                new_body.pop()  # drop last segment

            new_bodies.append(new_body)

            # Health: lose 1 each turn, reset to max if we ate
            snake.health -= 1
            if will_grow[i]:
                snake.health = self.max_health

        # Apply new bodies
        for i, snake in enumerate(self.snakes):
            snake.body = new_bodies[i]

        # 8) Starvation: any snake with health <= 0 dies
        for i, snake in enumerate(self.snakes):
            if snake.alive and snake.health <= 0:
                snake.alive = False
                rewards[i] -= 1.0

        # 9) Consume food for surviving snakes and respawn according to rules
        # Rebuild food list: remove any food that a *surviving* snake just stepped on
        new_food_positions: List[Tuple[int, int]] = []
        for fx, fy in self.food_positions:
            consumed = False
            for i, snake in enumerate(self.snakes):
                if not snake.alive or eliminated[i]:
                    continue
                if snake.body and snake.body[0] == (fx, fy):
                    consumed = True
                    snake.score += 1  # count food eaten
                    rewards[i] += 1.0  # positive reward for eating food
                    break
            if not consumed:
                new_food_positions.append((fx, fy))
        self.food_positions = new_food_positions

        # Ensure at least minimum_food is present, then maybe spawn extra
        self._ensure_minimum_food()
        self._maybe_spawn_extra_food()

        # 10) Rebuild grid from snake bodies/heads and food
        self._write_grid_from_state()

        # 11) Termination conditions
        if self.step_count >= self.max_steps:
            done = True

        if not any(s.alive for s in self.snakes):
            done = True

        # Build observations
        obs_n = self._build_all_obs()

        # 🔁 NEW: update GUI every step if in turtle mode
        if self.render_mode == "turtle":
            self._render_turtle()

        return obs_n, rewards, done, info

    def close(self) -> None:
        """Close any rendering resources."""
        if self.render_mode == "turtle" and self._screen is not None:
            try:
                turtle.bye()
            except Exception:
                pass
            self._screen = None
            self._drawer = None

    # ------------- Helpers -------------

    def _build_all_obs(self) -> List[np.ndarray]:
        return [self._build_obs_for_snake(i) for i in range(self.num_snakes)]

    def _write_grid_from_state(self) -> None:
        """Write food and snake bodies/heads into the grid for rendering."""
        self.grid.fill(0)

        # Food
        for fx, fy in self.food_positions:
            if self._in_bounds(fx, fy):
                self.grid[fy, fx] = 1

        # Snakes
        for i, snake in enumerate(self.snakes):
            if not snake.alive:
                continue
            # body
            for bx, by in snake.body:
                if not self._in_bounds(bx, by):
                    continue
                self.grid[by, bx] = 10 + i
            # head overwrites body
            hx, hy = snake.body[0]
            if self._in_bounds(hx, hy):
                self.grid[hy, hx] = 20 + i

    def _build_obs_for_snake(self, idx: int) -> np.ndarray:
        """Build the per-snake observation grid.

        0 = empty
        1 = food
        2 = own head
        3 = own body
        4 = opponent head/body
        """
        obs = np.zeros_like(self.grid, dtype=np.int8)

        # Mark all food locations as 1
        for fx, fy in self.food_positions:
            if self._in_bounds(fx, fy):
                obs[fy, fx] = 1

        # Mark snakes
        for j, snake in enumerate(self.snakes):
            if not snake.alive:
                continue
            # bodies (excluding head)
            for bx, by in snake.body[1:]:
                if not self._in_bounds(bx, by):
                    continue
                if j == idx:
                    obs[by, bx] = 3  # own body
                else:
                    obs[by, bx] = 4  # other body
            # heads
            hx, hy = snake.body[0]
            if not self._in_bounds(hx, hy):
                continue
            if j == idx:
                obs[hy, hx] = 2  # own head
            else:
                obs[hy, hx] = 4  # other head

        return obs

    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.grid_size and 0 <= y < self.grid_size

    def _center_pos(self) -> Tuple[int, int]:
        return self.grid_size // 2, self.grid_size // 2

    def _place_food(self) -> None:
        """Place a single new food on a random empty cell (no snakes, no existing food)."""
        empties: List[Tuple[int, int]] = []
        snake_cells = set()
        for s in self.snakes:
            if not s.alive:
                continue
            for x, y in s.body:
                snake_cells.add((x, y))

        food_cells = set(self.food_positions)

        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if (x, y) in snake_cells:
                    continue
                if (x, y) in food_cells:
                    continue
                empties.append((x, y))

        if not empties:
            return

        pos = random.choice(empties)
        self.food_positions.append(pos)

    def _ensure_minimum_food(self) -> None:
        """Guarantee at least minimum_food items on the board."""
        while len(self.food_positions) < self.minimum_food:
            self._place_food()

    def _maybe_spawn_extra_food(self) -> None:
        """Battlesnake-style: each turn we may spawn an extra food with some probability."""
        if self.food_spawn_chance <= 0:
            return
        if random.random() < (self.food_spawn_chance / 100.0):
            self._place_food()

    # ---------- Turtle rendering ----------

    def _init_turtle(self) -> None:
        if not _HAS_TURTLE:
            print("Turtle not available; render_mode will do nothing.")
            return
        if self._screen is None:
            turtle.colormode(255)
            self._screen = Screen()
            size_px = self.grid_size * self._cell_px
            self._screen.setup(width=size_px + 40, height=size_px + 40)
            self._screen.bgcolor("black")
            self._screen.title("SnakeGridEnv (turtle render)")
            self._screen.tracer(0)

            self._drawer = Turtle()
            self._drawer.hideturtle()
            self._drawer.penup()
            self._drawer.shape("square")
            self._drawer.shapesize(
                stretch_wid=self._cell_px / 20,
                stretch_len=self._cell_px / 20,
            )

        self._render_turtle()

    def _render_turtle(self) -> None:
        if self._screen is None or self._drawer is None:
            return

        self._drawer.clearstamps()

        g = self.grid_size
        half = (g * self._cell_px) / 2

        for y in range(g):
            for x in range(g):
                val = self.grid[y, x]
                if val == 0:
                    continue

                # Food
                if val == 1:
                    color = (255, 255, 0)
                # Snake heads
                elif 20 <= val < 30:
                    idx = val - 20
                    color = (255, 0, 255) if idx == 0 else (0, 255, 255)
                # Snake bodies
                else:
                    idx = val - 10
                    if idx == 0:
                        color = (180, 60, 200)
                    else:
                        color = (60, 200, 180)

                self._drawer.color(color)
                self._drawer.goto(
                    -half + x * self._cell_px + self._cell_px / 2,
                    half - y * self._cell_px - self._cell_px / 2,
                )
                self._drawer.stamp()

        self._screen.update()

    def render(self, mode: str = "turtle") -> None:
        """Compat wrapper: render current frame if using turtle."""
        if self.render_mode == "turtle":
            self._render_turtle()


if __name__ == "__main__":
    # Quick manual test: run a 2-snake game in turtle mode with random actions
    import time

    env = SnakeGridEnv(grid_size=11, num_snakes=2, render_mode="turtle", max_steps=200)

    obs_n = env.reset()
    done = False

    while not done:
        actions = [random.randint(0, 3) for _ in range(env.num_snakes)]
        obs_n, rew_n, done, info = env.step(actions)
        time.sleep(0.05)

    print("Episode finished. Close the turtle window to exit.")
    if _HAS_TURTLE:
        turtle.done()

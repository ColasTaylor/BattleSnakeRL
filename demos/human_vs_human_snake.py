import sys
import turtle
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from snake_rl.envs.snake_env import SnakeGridEnv  # noqa: E402

# Global state for the game loop
env = None
current_actions = [0, 0]  # actions for snake 0 and 1
paused = False
step_delay_ms = 150  # milliseconds between steps


def set_action(snake_idx: int, direction: int) -> None:
    """Update the desired direction for a given snake."""
    global current_actions
    current_actions[snake_idx] = direction


def toggle_pause() -> None:
    global paused
    paused = not paused
    print("[PAUSE]" if paused else "[RESUME]")


def game_step() -> None:
    """Single step of the environment driven by turtle's timer."""
    global env, paused

    # If screen is gone, stop
    if env._screen is None:
        return

    if paused:
        # Just schedule next check, no step
        env._screen.ontimer(game_step, step_delay_ms)
        return

    # Step the env with the current actions
    obs_n, reward_n, done, info = env.step(current_actions)

    # Print rewards each step if you want
    # print("Rewards:", reward_n)

    if done:
        print("Game over!")
        print("Final rewards:", reward_n)
        # Do NOT reschedule if you want it to stop here.
        # If you'd rather auto-restart, you could call reset() and reschedule.
        return

    # Schedule next step
    env._screen.ontimer(game_step, step_delay_ms)


def main() -> None:
    global env, current_actions

    # Create a 2-snake env in turtle mode.
    # Adjust grid_size / max_steps if you like.
    env = SnakeGridEnv(
        grid_size=11,
        num_snakes=2,
        render_mode="turtle",
        wrap=False,
        max_steps=500,
    )

    _ = env.reset()

    # Start actions as current directions (so they keep going straight)
    current_actions = [
        env.snakes[0].direction,
        env.snakes[1].direction,
    ]

    screen = env._screen
    if screen is None:
        print("Turtle screen not initialized; check snake_env.render_mode='turtle'.")
        return

    # IMPORTANT: in your DIRS, index→(dx,dy) is:
    # 0: (0,  1)  "up"   (grid y+; visually DOWN)
    # 1: (0, -1)  "down" (grid y-; visually UP)
    # 2: (-1, 0)  left
    # 3: (1,  0)  right
    #
    # We want key presses to correspond to **visual** directions:
    #   Up arrow   → visually UP  → direction 1
    #   Down arrow → visually DOWN→ direction 0
    #   Left arrow → direction 2
    #   Right arrow→ direction 3

    # Player 1 (Snake 0) – Arrow keys
    screen.listen()
    screen.onkey(lambda: set_action(0, 1), "Up")  # visually up
    screen.onkey(lambda: set_action(0, 0), "Down")  # visually down
    screen.onkey(lambda: set_action(0, 2), "Left")
    screen.onkey(lambda: set_action(0, 3), "Right")

    # Player 2 (Snake 1) – WASD
    # We'll keep the same visual mapping:
    #   W → up, S → down, A → left, D → right
    screen.onkey(lambda: set_action(1, 1), "w")
    screen.onkey(lambda: set_action(1, 0), "s")
    screen.onkey(lambda: set_action(1, 2), "a")
    screen.onkey(lambda: set_action(1, 3), "d")
    screen.onkey(lambda: set_action(1, 1), "W")
    screen.onkey(lambda: set_action(1, 0), "S")
    screen.onkey(lambda: set_action(1, 2), "A")
    screen.onkey(lambda: set_action(1, 3), "D")

    # Spacebar to pause / resume
    screen.onkey(toggle_pause, "space")

    # Kick off the timed game loop
    screen.ontimer(game_step, step_delay_ms)

    # Hand control over to turtle's main loop
    turtle.done()


if __name__ == "__main__":
    main()

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from snake_rl.eval.battle import make_agent  # noqa: E402
from snake_rl.envs.snake_grid_env import SnakeGridEnv  # noqa: E402


def main() -> None:
    """Watch a single agent play in the turtle-rendered environment."""
    # Which agent to watch: "heuristic", "ppo", or "dqn"
    agent_kind = sys.argv[1].lower() if len(sys.argv) > 1 else "heuristic"

    # Where to look for checkpoints for RL agents
    checkpoint = None
    if agent_kind == "dqn":
        checkpoint = "checkpoints/dqn_snake.pt"
    elif agent_kind == "ppo":
        #     checkpoint = "checkpoints/ppo_snake.pt"
        # elif agent_kind == "hmm":
        checkpoint = "checkpoints/ppo_hmm_two_snake.pt"

    # Single-snake grid env, but with turtle rendering enabled
    env = SnakeGridEnv(
        grid_size=15,
        num_snakes=1,
        render_mode="turtle",
        max_steps=500,
    )

    # Get one observation to infer obs_shape
    obs_n = env.reset()
    obs = obs_n[0]
    obs_shape = obs.shape
    num_actions = 4

    # Build the agent (heuristic / PPO / DQN)
    agent = make_agent(
        kind=agent_kind,
        obs_shape=obs_shape,
        num_actions=num_actions,
        checkpoint=checkpoint,
    )

    episode = 0
    try:
        while True:
            done = False
            ep_reward = 0.0

            while not done:
                # Deterministic action to actually see the learned/heuristic policy
                action = agent.act(obs, deterministic=True)
                next_obs_n, rewards, done, info = env.step(action)
                obs = next_obs_n[0]
                ep_reward += rewards[0]
                time.sleep(0.05)  # slow down so you can watch it

            print(
                f"Episode {episode} finished | agent={agent_kind} | reward={ep_reward:.2f}"
            )
            episode += 1

            # Auto-respawn for another round
            obs_n = env.reset()
            obs = obs_n[0]
    except KeyboardInterrupt:
        print("Stopping viewer.")
    finally:
        # Keep the turtle window open until closed by the user
        try:
            import turtle

            turtle.done()
        finally:
            env.close()


if __name__ == "__main__":
    main()

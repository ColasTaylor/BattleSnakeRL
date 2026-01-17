import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from snake_rl.eval.battle import make_agent  # noqa: E402
from snake_rl.envs.snake_grid_env import SnakeGridEnv  # noqa: E402
from snake_rl.opponent_model.hmm import HMMOpponentModel  # noqa: E402
# optional: import a heuristic or just use random
# from heuristic_agents import SafeSpaceHeuristicAgent


class RandomOpponent:
    """Random-action opponent for visualization."""

    def __init__(self, num_actions: int = 4) -> None:
        self.num_actions = num_actions

    def act(self, obs: np.ndarray) -> int:
        return np.random.randint(self.num_actions)


def main() -> None:
    """Visualize PPO+HMM agent against a random opponent."""
    # MUST match training settings for ppo_hmm_two_snake
    env = SnakeGridEnv(
        grid_size=11,  # training grid_size
        num_snakes=2,
        max_steps=500,
        render_mode="turtle",
    )

    hmm = HMMOpponentModel()

    # Reset env + HMM once to infer augmented obs dim
    obs_n = env.reset()
    hmm.reset(env, learning_idx=0, opponent_idx=1)

    base_obs = obs_n[0].astype(np.float32).flatten()
    belief = hmm.get_belief().astype(np.float32)
    obs_aug = np.concatenate([base_obs, belief], axis=0)
    obs_shape = obs_aug.shape  # (125,)

    num_actions = 4

    # Build PPO agent using the HMM-trained checkpoint
    agent = make_agent(
        kind="ppo",  # still "ppo", but with correct obs_shape
        obs_shape=obs_shape,
        num_actions=num_actions,
        checkpoint="checkpoints/ppo_hmm_two_snake.pt",
    )

    opponent = RandomOpponent(num_actions=num_actions)

    episode = 0

    try:
        while True:
            obs_n = env.reset()
            hmm.reset(env, learning_idx=0, opponent_idx=1)

            base_obs = obs_n[0].astype(np.float32).flatten()
            belief = hmm.get_belief().astype(np.float32)
            obs_aug = np.concatenate([base_obs, belief], axis=0)

            done = False
            ep_reward = 0.0

            while not done:
                # our snake (index 0) uses PPO+HMM
                a0 = agent.act(obs_aug, deterministic=True)

                # opponent (index 1) uses random or heuristic
                opp_obs = obs_n[1]
                a1 = opponent.act(opp_obs)

                next_obs_n, rewards, done, info = env.step([a0, a1])

                # update HMM with opponent's movement
                hmm.update_from_env(env)

                # build next augmented obs for snake 0
                base_obs = next_obs_n[0].astype(np.float32).flatten()
                belief = hmm.get_belief().astype(np.float32)
                obs_aug = np.concatenate([base_obs, belief], axis=0)

                obs_n = next_obs_n
                ep_reward += rewards[0]

                time.sleep(0.05)  # slow for visualization

            print(f"Episode {episode} finished | reward={ep_reward:.2f}")
            episode += 1

    except KeyboardInterrupt:
        print("Stopping viewer.")
    finally:
        try:
            import turtle

            turtle.done()
        finally:
            env.close()


if __name__ == "__main__":
    main()

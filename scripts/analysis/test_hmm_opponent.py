import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from snake_rl.agents.heuristic_agents import (  # noqa: E402
    GreedyFoodAgent,
    LoopyBotAgent,
    RandomAgent,
    RightBotAgent,
    SafeSpaceHeuristicAgent,
)
from snake_rl.envs.snake_env import SnakeGridEnv  # noqa: E402
from snake_rl.opponent_model.hmm_opponent_model import HMMOpponentModel  # noqa: E402

STYLE_NAMES = ["hungry", "scared", "loopy", "right"]


def make_opp_for_style(style: str, num_actions: int):
    """Construct the heuristic opponent for a named style."""
    style = style.lower()
    if style == "hungry":
        return GreedyFoodAgent(num_actions, name="hungry_bot")
    if style == "scared":
        return SafeSpaceHeuristicAgent(num_actions, name="scared_bot")
    if style == "loopy":
        return LoopyBotAgent(num_actions, name="loopy_bot")
    if style == "right":
        return RightBotAgent(num_actions, name="right_bot")
    raise ValueError(style)


def main() -> None:
    num_actions = 4

    for true_style in STYLE_NAMES:
        print(f"\n=== Testing HMM vs {true_style.upper()} bot ===")
        env = SnakeGridEnv(
            grid_size=11,
            num_snakes=2,
            max_steps=100,
            render_mode="none",
        )
        hmm = HMMOpponentModel(style_names=STYLE_NAMES)
        opponent = make_opp_for_style(true_style, num_actions=num_actions)
        random_me = RandomAgent(num_actions=num_actions, name="random_me")

        for ep in range(3):
            obs_n = env.reset()
            hmm.reset(env, learning_idx=0, opponent_idx=1)

            done = False
            step = 0

            while not done and step < 60:
                step += 1
                # snake 0 = us (random), snake 1 = style bot
                a0 = random_me.act(obs_n[0])
                a1 = opponent.act(obs_n[1])

                obs_n, rew_n, done, info = env.step([a0, a1])
                hmm.update_from_env(env)

            belief = hmm.get_belief()
            print(
                f"  ep {ep}: final belief={belief} | argmax={STYLE_NAMES[int(np.argmax(belief))]}"
            )

        env.close()


if __name__ == "__main__":
    main()

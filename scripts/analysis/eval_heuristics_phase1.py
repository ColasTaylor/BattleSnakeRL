"""
Evaluate all five heuristic snakes on the single-snake Battlesnake grid and
write per-episode metrics to separate CSV files. Metrics captured:
- episodic return (performance)
- episode length in steps (survival)
- food eaten (food collection)

Use this to generate horizontal reference lines for heuristics alongside PPO/DQN
learning curves in Phase 1 analysis.
"""

import argparse
import csv
import os
import random
import sys
from typing import Callable, Dict, List, Sequence
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from snake_rl.agents.heuristics import (  # noqa: E402
    RandomAgent,
    GreedyFoodAgent,
    SafeSpaceHeuristicAgent,
    LoopyBotAgent,
    RightBotAgent,
)
from snake_rl.envs.snake_grid_env import SnakeGridEnv  # noqa: E402


def set_seed(seed: int) -> None:
    """Seed Python and NumPy RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def evaluate_heuristic(
    agent_name: str,
    agent_ctor: Callable[..., object],
    seeds: Sequence[int],
    episodes_per_seed: int,
    grid_size: int,
    max_steps: int,
    log_dir: str,
) -> Dict[str, float]:
    """Run episodes for a heuristic agent; return aggregate stats and write a CSV.
    CSV columns: seed, episode, reward, steps, food_eaten.
    """

    os.makedirs(log_dir, exist_ok=True)
    csv_path = os.path.join(log_dir, f"{agent_name}.csv")

    rows: List[Dict[str, float]] = []
    rewards: List[float] = []
    foods: List[float] = []
    steps_list: List[int] = []

    for seed in seeds:
        set_seed(seed)
        env = SnakeGridEnv(
            grid_size=grid_size,
            num_snakes=1,
            render_mode="none",
            max_steps=max_steps,
        )

        for ep in range(episodes_per_seed):
            # fresh agent each episode (some heuristics keep internal state)
            agent = agent_ctor(num_actions=4, name=agent_name)

            obs = env.reset()[0]
            done = False
            ep_reward = 0.0
            ep_steps = 0

            while not done:
                action = agent.act(obs, deterministic=True)
                next_obs_n, rewards_n, done, _ = env.step(action)
                obs = next_obs_n[0]
                ep_reward += rewards_n[0]
                ep_steps += 1

            food_eaten = env.snakes[0].score if env.snakes else 0

            rows.append(
                {
                    "seed": seed,
                    "episode": ep,
                    "reward": ep_reward,
                    "steps": ep_steps,
                    "food_eaten": food_eaten,
                }
            )
            rewards.append(ep_reward)
            foods.append(food_eaten)
            steps_list.append(ep_steps)

        env.close()

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["seed", "episode", "reward", "steps", "food_eaten"]
        )
        writer.writeheader()
        writer.writerows(rows)

    stats: Dict[str, float] = {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_steps": float(np.mean(steps_list)),
        "std_steps": float(np.std(steps_list)),
        "mean_food": float(np.mean(foods)),
        "std_food": float(np.std(foods)),
        "csv_path": csv_path,
    }
    return stats


def parse_seeds(seed_str: str) -> List[int]:
    """Parse a comma-separated seed string into integers."""
    return [int(s) for s in seed_str.split(",") if s.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate heuristic snakes for Phase 1"
    )
    parser.add_argument(
        "--seeds", type=str, default="0,1,2,3,4", help="Comma-separated seeds"
    )
    parser.add_argument("--episodes_per_seed", type=int, default=20)
    parser.add_argument("--grid_size", type=int, default=15)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--log_dir", type=str, default="logs/phase1_heuristics")
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)

    heuristics = [
        ("random", RandomAgent),
        ("greedy_food", GreedyFoodAgent),
        ("heuristic_safe", SafeSpaceHeuristicAgent),
        ("loopy_bot", LoopyBotAgent),
        ("right_bot", RightBotAgent),
    ]

    all_stats: Dict[str, Dict[str, float]] = {}
    for name, ctor in heuristics:
        stats = evaluate_heuristic(
            agent_name=name,
            agent_ctor=ctor,
            seeds=seeds,
            episodes_per_seed=args.episodes_per_seed,
            grid_size=args.grid_size,
            max_steps=args.max_steps,
            log_dir=args.log_dir,
        )
        all_stats[name] = stats
        print(
            f"{name}: mean_reward={stats['mean_reward']:.3f} ",
            f"mean_steps={stats['mean_steps']:.2f} ",
            f"mean_food={stats['mean_food']:.2f} | csv={stats['csv_path']}",
        )

    print("\nSaved per-heuristic CSVs to:")
    for name, stats in all_stats.items():
        print(f"  {name}: {stats['csv_path']}")


if __name__ == "__main__":
    main()

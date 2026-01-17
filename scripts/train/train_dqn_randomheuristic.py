import argparse
import csv
import os
import random
import sys
import time
from collections import deque
from pathlib import Path
from typing import Deque, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from snake_rl.agents.dqn import DQNAgent  # noqa: E402
from snake_rl.agents.heuristics import (  # noqa: E402
    RandomAgent,
    GreedyFoodAgent,
    SafeSpaceHeuristicAgent,
    LoopyBotAgent,
    RightBotAgent,
)
from snake_rl.envs.snake_grid_env import SnakeGridEnv  # noqa: E402


def train_dqn_randomheuristic(
    total_steps: int = 350_000,
    grid_size: int = 15,
    log_path: str = "logs/dqn_randomheuristic_train.csv",
    save_path: Optional[str] = "models/dqn_randomheuristic.pt",
) -> None:
    """Train a DQN agent against a rotating set of heuristic opponents."""
    env = SnakeGridEnv(
        grid_size=grid_size, num_snakes=2, render_mode="none", max_steps=500
    )
    obs_n = env.reset()
    obs_shape = obs_n[0].shape
    num_actions = 4

    agent = DQNAgent(
        obs_shape=obs_shape,
        num_actions=num_actions,
        name="dqn_randomheuristic",
    )

    def sample_opponent() -> Tuple[object, str]:
        options = [
            ("random", RandomAgent),
            ("greedy_food", GreedyFoodAgent),
            ("heuristic_safe", SafeSpaceHeuristicAgent),
            ("loopy_bot", LoopyBotAgent),
            ("right_bot", RightBotAgent),
        ]
        name, ctor = random.choice(options)
        return ctor(num_actions=num_actions, name=name), name

    step_count = 0
    episode = 0
    recent_rewards: Deque[float] = deque(maxlen=50)

    writer = None
    log_f = None
    try:
        if log_path:
            os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
            log_f = open(log_path, "w", newline="")
            writer = csv.writer(log_f)
            writer.writerow(
                ["episode", "total_steps", "ep_reward", "ma_reward", "opponent"]
            )
    except PermissionError as e:
        print(f"[WARN] Could not open log file {log_path}: {e}. Logging disabled.")

    start_time = time.time()
    try:
        while step_count < total_steps:
            episode += 1
            obs_n = env.reset()
            done = False
            ep_reward = 0.0

            learn_idx = 0
            opp_idx = 1
            opponent, opp_name = sample_opponent()

            while not done:
                actions = [0 for _ in range(env.num_snakes)]

                # Learning snake: epsilon-greedy, recorded
                a_learn = agent.act(obs_n[learn_idx], deterministic=False)
                # Opponent snake: heuristic policy
                a_opp = opponent.act(obs_n[opp_idx], deterministic=True)

                actions[learn_idx] = a_learn
                actions[opp_idx] = a_opp

                next_obs_n, rewards, done, info = env.step(actions)

                transition = {
                    "obs": obs_n[learn_idx],
                    "action": a_learn,
                    "reward": rewards[learn_idx],
                    "next_obs": next_obs_n[learn_idx],
                    "done": done,
                }
                agent.observe(transition)
                update_stats = agent.update()
                if update_stats and episode % 50 == 0:
                    s = ", ".join(f"{k}={v:.2f}" for k, v in update_stats.items())
                    print(f"[DQN RANDOM-HEUR] Ep={episode} | steps={step_count} | {s}")

                step_count += 1
                ep_reward += rewards[learn_idx]
                obs_n = next_obs_n

                if step_count >= total_steps:
                    break

            recent_rewards.append(ep_reward)
            ma_reward = float(np.mean(recent_rewards))
            print(
                f"[TRAIN RANDOM-HEUR] Ep={episode:4d} | steps={step_count:7d} | "
                f"ep_reward={ep_reward:6.2f} | ma_reward(50)={ma_reward:6.2f} | opp={opp_name}"
            )
            if writer:
                writer.writerow([episode, step_count, ep_reward, ma_reward, opp_name])

        if save_path:
            save_dir = os.path.dirname(save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            agent.save(save_path)
        elapsed = time.time() - start_time
        print(f"DQN random-heuristic training finished in {elapsed:.1f} seconds")
    finally:
        if log_f:
            log_f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--total_steps", type=int, default=350_000)
    parser.add_argument("--grid_size", type=int, default=15)
    parser.add_argument(
        "--log_path", type=str, default="logs/dqn_randomheuristic_train.csv"
    )
    parser.add_argument(
        "--save_path", type=str, default="models/dqn_randomheuristic.pt"
    )
    args = parser.parse_args()

    train_dqn_randomheuristic(
        total_steps=args.total_steps,
        grid_size=args.grid_size,
        log_path=args.log_path,
        save_path=args.save_path,
    )

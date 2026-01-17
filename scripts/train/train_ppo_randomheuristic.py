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

from snake_rl.agents.heuristics import (  # noqa: E402
    RandomAgent,
    GreedyFoodAgent,
    SafeSpaceHeuristicAgent,
    LoopyBotAgent,
    RightBotAgent,
)
from snake_rl.agents.ppo import PPOAgent  # noqa: E402
from snake_rl.envs.snake_grid_env import SnakeGridEnv  # noqa: E402


def train_ppo_randomheuristic(
    total_steps: int = 350_000,
    rollout_length: int = 2_048,
    grid_size: int = 15,
    log_path: str = "logs_ppo_hmm/ppo_randomheuristic_train.csv",
    save_path: Optional[str] = "models/ppo_randomheuristic.pt",
    lr: float = 3e-4,
    entropy_coef: float = 0.0,
    clip_range: float = 0.3,
    gae_lambda: float = 0.95,
    num_epochs: int = 4,
    init_checkpoint: Optional[str] = None,
) -> None:
    """Train PPO against a rotating set of heuristic opponents."""
    # Two-snake environment
    env = SnakeGridEnv(
        grid_size=grid_size, num_snakes=2, render_mode="none", max_steps=500
    )
    obs_n = env.reset()
    obs_shape = obs_n[0].shape
    num_actions = 4

    agent = PPOAgent(
        obs_shape=obs_shape,
        num_actions=num_actions,
        rollout_length=rollout_length,
        name="ppo_randomheuristic",
        lr=lr,
        entropy_coef=entropy_coef,
        clip_eps=clip_range,
        lam=gae_lambda,
        num_epochs=num_epochs,
    )
    # Optional warm-start from a 1-snake PPO checkpoint
    if init_checkpoint:
        try:
            agent.load(init_checkpoint)
            print(f"[INIT] Loaded PPO weights from {init_checkpoint}")
        except Exception as e:
            print(f"[WARN] Could not load init checkpoint {init_checkpoint}: {e}")

    def sample_opponent() -> Tuple[object, str]:
        """Pick a random heuristic opponent each episode."""
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
            obs_n = env.reset()
            done = False
            ep_reward = 0.0

            # Fixed learning snake = index 0; index 1 = heuristic opponent (random each episode)
            learn_idx = 0
            opp_idx = 1
            opponent, opp_name = sample_opponent()

            while not done:
                actions = [0 for _ in range(env.num_snakes)]

                # Learning snake: stochastic, recorded to PPO buffer
                actions[learn_idx] = agent.act(obs_n[learn_idx], deterministic=False)

                # Opponent snake: heuristic, deterministic
                actions[opp_idx] = opponent.act(obs_n[opp_idx], deterministic=True)

                next_obs_n, rewards, done, info = env.step(actions)

                # Only learn from the chosen snake
                agent.observe({"reward": rewards[learn_idx], "done": done})

                step_count += 1
                ep_reward += rewards[learn_idx]
                obs_n = next_obs_n

                if step_count % agent.rollout_length == 0:
                    update_stats = agent.update()
                    if update_stats:
                        print(
                            f"[steps={step_count}] PPO random-heuristic update: "
                            + ", ".join(f"{k}={v:.4f}" for k, v in update_stats.items())
                        )

                if step_count >= total_steps:
                    break

            episode += 1
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
        print(f"PPO random-heuristic training finished in {elapsed:.1f} seconds")
    finally:
        if log_f:
            log_f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--total_steps", type=int, default=400_000)
    parser.add_argument("--rollout_length", type=int, default=2_048)
    parser.add_argument("--grid_size", type=int, default=15)
    parser.add_argument(
        "--log_path", type=str, default="logs_ppo_hmm/ppo_randomheuristic_train.csv"
    )
    parser.add_argument(
        "--save_path", type=str, default="models/ppo_randomheuristic.pt"
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--entropy_coef", type=float, default=0.0)
    parser.add_argument("--clip_range", type=float, default=0.3)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--num_epochs", type=int, default=4)
    parser.add_argument("--init_checkpoint", type=str, default=None)
    args = parser.parse_args()

    train_ppo_randomheuristic(
        total_steps=args.total_steps,
        rollout_length=args.rollout_length,
        grid_size=args.grid_size,
        log_path=args.log_path,
        save_path=args.save_path,
        lr=args.lr,
        entropy_coef=args.entropy_coef,
        clip_range=args.clip_range,
        gae_lambda=args.gae_lambda,
        num_epochs=args.num_epochs,
        init_checkpoint=args.init_checkpoint,
    )

import csv
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from snake_rl.agents.dqn import DQNAgent  # noqa: E402
from snake_rl.envs.snake_grid_env import SnakeGridEnv  # noqa: E402


def evaluate_dqn(
    agent: DQNAgent,
    grid_size: int = 15,
    num_episodes: int = 10,
    max_steps: int = 500,
) -> Dict[str, float]:
    """Run evaluation episodes with a deterministic policy (no exploration, no learning).
    Returns mean and std of episodic rewards.
    """
    eval_env = SnakeGridEnv(
        grid_size=grid_size,
        num_snakes=1,
        render_mode="none",
        max_steps=max_steps,
    )
    ep_rewards = []
    ep_foods = []

    for _ in range(num_episodes):
        obs_n = eval_env.reset()
        obs = obs_n[0]
        done = False
        ep_ret = 0.0
        while not done:
            action = agent.act(obs, deterministic=True)
            next_obs_n, rewards, done, info = eval_env.step(action)
            obs = next_obs_n[0]
            ep_ret += rewards[0]
        ep_rewards.append(ep_ret)
        ep_foods.append(eval_env.snakes[0].score if eval_env.snakes else 0)

    eval_env.close()
    ep_rewards = np.array(ep_rewards, dtype=np.float32)
    ep_foods = np.array(ep_foods, dtype=np.float32)
    return {
        "eval_mean_reward": float(ep_rewards.mean()),
        "eval_std_reward": float(ep_rewards.std()),
        "eval_mean_food": float(ep_foods.mean()),
        "eval_std_food": float(ep_foods.std()),
    }


def train_dqn_single(
    total_steps: int = 200_000,
    grid_size: int = 15,
    max_steps_per_ep: int = 500,
    eval_every_episodes: int = 50,
    log_dir: str = "logs",
    save_path: Optional[str] = None,
) -> Tuple[DQNAgent, SnakeGridEnv]:
    """Train a DQN agent on the single-snake environment."""
    env = SnakeGridEnv(
        grid_size=grid_size,
        num_snakes=1,
        render_mode="none",
        max_steps=max_steps_per_ep,
    )

    # Initial obs to get shape
    obs_n = env.reset()
    obs = obs_n[0]
    obs_shape = obs.shape
    num_actions = 4

    agent = DQNAgent(
        obs_shape=obs_shape,
        num_actions=num_actions,
    )

    step_count = 0
    episode = 0

    # For moving average of training rewards
    recent_rewards: Deque[float] = deque(maxlen=50)

    train_writer = eval_writer = None
    train_log_f = eval_log_f = None
    try:
        os.makedirs(log_dir, exist_ok=True)
        train_log_f = open(os.path.join(log_dir, "dqn_train.csv"), "w", newline="")
        train_writer = csv.writer(train_log_f)
        train_writer.writerow(
            ["episode", "total_steps", "ep_reward", "ep_food", "ma_reward"]
        )

        eval_log_f = open(os.path.join(log_dir, "dqn_eval.csv"), "w", newline="")
        eval_writer = csv.writer(eval_log_f)
        eval_writer.writerow(
            [
                "episode",
                "total_steps",
                "eval_mean_reward",
                "eval_std_reward",
                "eval_mean_food",
                "eval_std_food",
            ]
        )
    except PermissionError as e:
        print(f"[WARN] Could not open log files in {log_dir}: {e}. Logging disabled.")

    while step_count < total_steps:
        obs_n = env.reset()
        obs = obs_n[0]
        done = False
        ep_reward = 0.0
        ep_food = 0

        while not done:
            # Training action (epsilon-greedy)
            action = agent.act(obs, deterministic=False)
            next_obs_n, rewards, done, info = env.step(action)
            next_obs = next_obs_n[0]
            reward = rewards[0]

            # Store transition
            agent.observe(
                {
                    "obs": obs,
                    "action": action,
                    "reward": reward,
                    "next_obs": next_obs,
                    "done": done,
                }
            )

            # Training update
            _ = agent.update()  # may return {"dqn_loss": ...} after warmup

            ep_reward += reward
            obs = next_obs
            step_count += 1

            if step_count >= total_steps:
                break

        episode += 1
        ep_food = env.snakes[0].score if env.snakes else 0
        recent_rewards.append(ep_reward)
        ma_reward = float(np.mean(recent_rewards))

        # Basic training log
        print(
            f"[TRAIN] Ep={episode:4d} | steps={step_count:7d} | "
            f"ep_reward={ep_reward:6.2f} | ep_food={ep_food:3d} | ma_reward(50)={ma_reward:6.2f}"
        )

        if train_writer:
            train_writer.writerow([episode, step_count, ep_reward, ep_food, ma_reward])

        # Periodic evaluation (no exploration, no learning)
        if episode % eval_every_episodes == 0:
            eval_stats = evaluate_dqn(
                agent,
                grid_size=grid_size,
                num_episodes=10,
                max_steps=max_steps_per_ep,
            )
            print(
                f"[EVAL ] Ep={episode:4d} | steps={step_count:7d} | "
                f"mean_reward={eval_stats['eval_mean_reward']:6.2f} | "
                f"std={eval_stats['eval_std_reward']:5.2f} | "
                f"mean_food={eval_stats['eval_mean_food']:5.2f} | "
                f"food_std={eval_stats['eval_std_food']:5.2f}"
            )
            if eval_writer:
                eval_writer.writerow(
                    [
                        episode,
                        step_count,
                        eval_stats["eval_mean_reward"],
                        eval_stats["eval_std_reward"],
                        eval_stats["eval_mean_food"],
                        eval_stats["eval_std_food"],
                    ]
                )

    if train_log_f:
        train_log_f.close()
    if eval_log_f:
        eval_log_f.close()

    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        agent.save(save_path)

    return agent, env


if __name__ == "__main__":
    start = time.time()
    agent, env = train_dqn_single(
        total_steps=350_000,
        grid_size=15,
        max_steps_per_ep=500,
        eval_every_episodes=50,
        log_dir="logs",
        save_path="checkpoints/dqn_snake.pt",  # <--- important
    )
    print("Training finished in {:.1f} seconds".format(time.time() - start))

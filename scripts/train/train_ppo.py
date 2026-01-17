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

from snake_rl.agents.ppo import PPOAgent  # noqa: E402
from snake_rl.envs.snake_grid_env import SnakeGridEnv  # noqa: E402


def evaluate_policy(
    agent: PPOAgent,
    grid_size: int = 15,
    num_episodes: int = 10,
    max_steps: int = 500,
) -> Dict[str, float]:
    """Run deterministic eval episodes for PPO (no learning)."""
    eval_env = SnakeGridEnv(
        grid_size=grid_size, num_snakes=1, render_mode="none", max_steps=max_steps
    )
    ep_rewards = []
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
    eval_env.close()
    ep_rewards = np.array(ep_rewards, dtype=np.float32)
    return {
        "eval_mean_reward": float(ep_rewards.mean()),
        "eval_std_reward": float(ep_rewards.std()),
    }


def train_ppo_single(
    total_steps: int = 200_000,
    rollout_length: int = 1024,
    grid_size: int = 15,
    eval_every_episodes: int = 50,
    log_dir: str = "logs",
    save_path: Optional[str] = None,
) -> Tuple[PPOAgent, SnakeGridEnv]:
    """Train a PPO agent on the single-snake environment."""
    env = SnakeGridEnv(
        grid_size=grid_size, num_snakes=1, render_mode="none", max_steps=500
    )
    obs_n = env.reset()
    obs = obs_n[0]
    obs_shape = obs.shape
    num_actions = 4
    agent = PPOAgent(
        obs_shape=obs_shape,
        num_actions=num_actions,
        rollout_length=rollout_length,
    )
    step_count = 0
    episode = 0
    recent_rewards: Deque[float] = deque(maxlen=50)

    train_writer = eval_writer = None
    train_log_f = eval_log_f = None
    try:
        os.makedirs(log_dir, exist_ok=True)
        train_log_f = open(os.path.join(log_dir, "ppo_train.csv"), "w", newline="")
        train_writer = csv.writer(train_log_f)
        train_writer.writerow(["episode", "total_steps", "ep_reward", "ma_reward"])

        eval_log_f = open(os.path.join(log_dir, "ppo_eval.csv"), "w", newline="")
        eval_writer = csv.writer(eval_log_f)
        eval_writer.writerow(
            ["episode", "total_steps", "eval_mean_reward", "eval_std_reward"]
        )
    except PermissionError as e:
        print(f"[WARN] Could not open log files in {log_dir}: {e}. Logging disabled.")

    while step_count < total_steps:
        obs_n = env.reset()
        obs = obs_n[0]
        done = False
        ep_reward = 0.0
        while not done:
            action = agent.act(obs, deterministic=False)
            next_obs_n, rewards, done, info = env.step(action)
            next_obs = next_obs_n[0]
            reward = rewards[0]
            agent.observe({"reward": reward, "done": done})
            ep_reward += reward
            obs = next_obs
            step_count += 1
            if step_count % agent.rollout_length == 0:
                update_stats = agent.update()
                if update_stats:
                    print(f"[steps={step_count}] PPO update: {update_stats}")
            if step_count >= total_steps:
                break
        episode += 1
        recent_rewards.append(ep_reward)
        ma_reward = float(np.mean(recent_rewards))
        print(
            f"[TRAIN] Ep={episode:4d} | steps={step_count:7d} | "
            f"ep_reward={ep_reward:6.2f} | ma_reward(50)={ma_reward:6.2f}"
        )
        if train_writer:
            train_writer.writerow([episode, step_count, ep_reward, ma_reward])

        if episode % eval_every_episodes == 0:
            eval_stats = evaluate_policy(
                agent,
                grid_size=grid_size,
                num_episodes=10,
                max_steps=500,
            )
            print(
                f"[EVAL ] Ep={episode:4d} | steps={step_count:7d} | "
                f"mean_reward={eval_stats['eval_mean_reward']:6.2f} | "
                f"std={eval_stats['eval_std_reward']:5.2f}"
            )
            if eval_writer:
                eval_writer.writerow(
                    [
                        episode,
                        step_count,
                        eval_stats["eval_mean_reward"],
                        eval_stats["eval_std_reward"],
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
    agent, env = train_ppo_single(
        total_steps=350_000,
        rollout_length=1024,
        grid_size=15,
        eval_every_episodes=50,
        log_dir="logs",
        save_path="checkpoints/ppo_snake.pt",  # <--- important
    )
    print("Training finished in {:.1f} seconds".format(time.time() - start))

    # Optional visualization of the trained PPO policy in turtle mode
    vis_env = SnakeGridEnv(
        grid_size=env.grid_size,
        num_snakes=1,
        render_mode="turtle",
        max_steps=300,
    )
    for ep in range(3):
        obs_n = vis_env.reset()
        obs = obs_n[0]
        done = False
        ep_reward = 0.0
        while not done:
            action = agent.act(obs, deterministic=True)
            obs_n, rewards, done, info = vis_env.step(action)
            obs = obs_n[0]
            ep_reward += rewards[0]
        print(f"[VIS] Episode {ep}: reward={ep_reward:.2f}")
    vis_env.close()

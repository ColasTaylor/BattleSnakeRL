import csv
import os
import random
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from snake_rl.agents.dqn_agent import DQNAgent
from snake_rl.agents.heuristic_agents import (
    GreedyFoodAgent,
    RandomAgent,
    SafeSpaceHeuristicAgent,
)
from snake_rl.agents.ppo_agent import PPOAgent
from snake_rl.envs.snake_env import SnakeGridEnv

try:
    import torch

    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


def set_seed(seed: int) -> None:
    """Set RNG seeds for reproducible evaluation."""
    random.seed(seed)
    np.random.seed(seed)
    if _HAS_TORCH:
        torch.manual_seed(seed)


def make_agent(
    kind: str,
    obs_shape: Sequence[int],
    num_actions: int,
    checkpoint: Optional[str] = None,
) -> object:
    """Factory for evaluation agents."""
    kind = kind.lower()
    if kind == "dqn":
        agent = DQNAgent(obs_shape=obs_shape, num_actions=num_actions)
        if checkpoint:
            agent.load(checkpoint)
        return agent
    if kind == "ppo":
        agent = PPOAgent(obs_shape=obs_shape, num_actions=num_actions)
        if checkpoint:
            agent.load(checkpoint)
        return agent
    if kind == "heuristic":
        return SafeSpaceHeuristicAgent(num_actions=num_actions)
    if kind == "greedy":
        return GreedyFoodAgent(num_actions=num_actions)
    if kind == "random":
        return RandomAgent(num_actions=num_actions)
    raise ValueError(f"Unknown agent kind: {kind}")


def evaluate_single_agent(
    agent_kind: str,
    obs_shape: Sequence[int],
    num_actions: int,
    seeds: Sequence[int],
    episodes_per_seed: int = 5,
    grid_size: int = 15,
    max_steps: int = 500,
    checkpoint: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Evaluate one agent across seeds and return per-episode metrics."""
    agent = make_agent(agent_kind, obs_shape, num_actions, checkpoint=checkpoint)
    results: List[Dict[str, Any]] = []

    for seed in seeds:
        set_seed(seed)
        env = SnakeGridEnv(
            grid_size=grid_size, num_snakes=1, render_mode="none", max_steps=max_steps
        )
        for ep in range(episodes_per_seed):
            obs_n = env.reset()
            obs = obs_n[0]
            done = False
            ep_reward = 0.0
            steps = 0
            while not done:
                action = agent.act(obs, deterministic=True)
                next_obs_n, rewards, done, info = env.step(action)
                obs = next_obs_n[0]
                ep_reward += rewards[0]
                steps += 1
            food_eaten = env.snakes[0].score if env.snakes else 0
            results.append(
                {
                    "agent": agent_kind,
                    "seed": seed,
                    "episode": ep,
                    "reward": ep_reward,
                    "food_eaten": food_eaten,
                    "steps": steps,
                }
            )
        env.close()
    return results


def run_evaluation(
    agent_specs: List[Dict[str, Any]],
    seeds: Sequence[int],
    episodes_per_seed: int = 5,
    grid_size: int = 15,
    max_steps: int = 500,
    log_dir: str = "logs",
    log_name: str = "eval_results.csv",
) -> None:
    """Evaluate multiple agents and write results to CSV."""
    # Get obs shape once
    probe_env = SnakeGridEnv(
        grid_size=grid_size, num_snakes=1, render_mode="none", max_steps=max_steps
    )
    obs_shape = probe_env.reset()[0].shape
    num_actions = 4
    probe_env.close()

    all_rows: List[Dict[str, Any]] = []
    for spec in agent_specs:
        kind = spec["kind"]
        checkpoint = spec.get("checkpoint")
        all_rows.extend(
            evaluate_single_agent(
                agent_kind=kind,
                obs_shape=obs_shape,
                num_actions=num_actions,
                seeds=seeds,
                episodes_per_seed=episodes_per_seed,
                grid_size=grid_size,
                max_steps=max_steps,
                checkpoint=checkpoint,
            )
        )

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_name)
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["agent", "seed", "episode", "reward", "food_eaten", "steps"]
        )
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)
    print(f"Wrote eval results to {log_path}")


if __name__ == "__main__":
    # Default setup: evaluate heuristic plus any checkpoints that exist.
    seeds = [0, 1, 2, 3, 4]
    agent_specs: List[Dict[str, Any]] = []

    # Include checkpoints if present
    dqn_ckpt = "checkpoints/dqn_snake.pt"
    ppo_ckpt = "checkpoints/ppo_snake.pt"
    if os.path.exists(dqn_ckpt):
        agent_specs.append({"kind": "dqn", "checkpoint": dqn_ckpt})
    if os.path.exists(ppo_ckpt):
        agent_specs.append({"kind": "ppo", "checkpoint": ppo_ckpt})

    # Always include heuristic baseline
    agent_specs.append({"kind": "heuristic"})

    run_evaluation(
        agent_specs=agent_specs,
        seeds=seeds,
        episodes_per_seed=5,
        grid_size=15,
        max_steps=500,
        log_dir="logs",
        log_name="eval_results.csv",
    )

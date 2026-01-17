import csv
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from snake_rl.agents.base_agent import Agent  # noqa: E402
from snake_rl.agents.dqn_agent import DQNAgent  # noqa: E402
from snake_rl.agents.heuristic_agents import (  # noqa: E402
    GreedyFoodAgent,
    RandomAgent,
    SafeSpaceHeuristicAgent,
)
from snake_rl.agents.ppo_agent import PPOAgent  # noqa: E402
from snake_rl.envs.snake_env import SnakeGridEnv  # noqa: E402


def make_agent(
    kind: str,
    obs_shape: Sequence[int],
    num_actions: int,
    name: str,
    checkpoint: Optional[str] = None,
) -> Agent:
    """Factory for battle agents by name."""
    kind = kind.lower()
    if kind == "random":
        return RandomAgent(num_actions=num_actions, name=name)
    if kind == "greedy":
        return GreedyFoodAgent(num_actions=num_actions, name=name)
    if kind == "heuristic":
        # Single- or multi-snake safe-space heuristic
        return SafeSpaceHeuristicAgent(num_actions=num_actions, name=name)
    if kind == "ppo":
        agent = PPOAgent(obs_shape=obs_shape, num_actions=num_actions, name=name)
        if checkpoint and os.path.exists(checkpoint):
            agent.load(checkpoint)
        return agent
    if kind == "dqn":
        agent = DQNAgent(obs_shape=obs_shape, num_actions=num_actions, name=name)
        if checkpoint and os.path.exists(checkpoint):
            agent.load(checkpoint)
        return agent
    raise ValueError(f"Unknown agent kind: {kind}")


def run_battle(
    agent1_kind: str = "ppo",
    agent2_kind: str = "greedy",
    num_episodes: int = 10,
    grid_size: int = 15,
    render: bool = True,
    checkpoint1: Optional[str] = None,
    checkpoint2: Optional[str] = None,
    log_csv: Optional[str] = None,
) -> Dict[str, Any]:
    """Run head-to-head matches between two agents and return summary stats."""
    render_mode = "turtle" if render else "none"
    env = SnakeGridEnv(
        grid_size=grid_size, num_snakes=2, render_mode=render_mode, max_steps=500
    )
    obs_n = env.reset()
    obs_shape = obs_n[0].shape
    num_actions = 4
    agent1 = make_agent(
        agent1_kind, obs_shape, num_actions, name="agent1", checkpoint=checkpoint1
    )
    agent2 = make_agent(
        agent2_kind, obs_shape, num_actions, name="agent2", checkpoint=checkpoint2
    )
    results = {
        "agent1_wins": 0,
        "agent2_wins": 0,
        "draws": 0,
        "mean_reward_agent1": 0.0,
        "mean_reward_agent2": 0.0,
        "mean_food_agent1": 0.0,
        "mean_food_agent2": 0.0,
        "mean_steps": 0.0,
    }

    writer = None
    log_f = None
    if log_csv:
        try:
            os.makedirs(os.path.dirname(log_csv) or ".", exist_ok=True)
            log_f = open(log_csv, "w", newline="")

            writer = csv.writer(log_f)
            writer.writerow(
                [
                    "episode",
                    "steps",
                    "reward_agent1",
                    "reward_agent2",
                    "food_agent1",
                    "food_agent2",
                    "alive_agent1",
                    "alive_agent2",
                    "outcome",
                ]
            )
        except PermissionError as e:
            print(f"[WARN] Could not open log file {log_csv}: {e}. Logging disabled.")

    sum_r1 = sum_r2 = 0.0
    sum_f1 = sum_f2 = 0.0
    sum_steps = 0
    for ep in range(num_episodes):
        obs_n = env.reset()
        done = False
        ep_rewards = [0.0, 0.0]
        steps = 0
        while not done:
            a1 = agent1.act(obs_n[0], deterministic=True)
            a2 = agent2.act(obs_n[1], deterministic=True)
            next_obs_n, rewards, done, info = env.step([a1, a2])
            obs_n = next_obs_n
            ep_rewards[0] += rewards[0]
            ep_rewards[1] += rewards[1]
            steps += 1
        if ep_rewards[0] > ep_rewards[1]:
            results["agent1_wins"] += 1
            outcome = "agent1 wins"
        elif ep_rewards[1] > ep_rewards[0]:
            results["agent2_wins"] += 1
            outcome = "agent2 wins"
        else:
            results["draws"] += 1
            outcome = "draw"
        food1 = env.snakes[0].score if env.snakes else 0
        food2 = env.snakes[1].score if len(env.snakes) > 1 else 0
        alive1 = int(env.snakes[0].alive) if env.snakes else 0
        alive2 = int(env.snakes[1].alive) if len(env.snakes) > 1 else 0
        sum_r1 += ep_rewards[0]
        sum_r2 += ep_rewards[1]
        sum_f1 += food1
        sum_f2 += food2
        sum_steps += steps
        if writer:
            writer.writerow(
                [
                    ep,
                    steps,
                    ep_rewards[0],
                    ep_rewards[1],
                    food1,
                    food2,
                    alive1,
                    alive2,
                    outcome,
                ]
            )
        print(
            f"Episode {ep}: r1={ep_rewards[0]:.2f}, r2={ep_rewards[1]:.2f}, food1={food1}, food2={food2} -> {outcome}"
        )
    if log_f:
        log_f.close()
    n = max(1, num_episodes)
    results["mean_reward_agent1"] = sum_r1 / n
    results["mean_reward_agent2"] = sum_r2 / n
    results["mean_food_agent1"] = sum_f1 / n
    results["mean_food_agent2"] = sum_f2 / n
    results["mean_steps"] = sum_steps / n
    print("=== Battle results ===")
    print(results)
    env.close()
    return results


if __name__ == "__main__":
    # Example: PPO vs greedy heuristic
    run_battle(agent1_kind="ppo", agent2_kind="greedy", num_episodes=5, render=True)

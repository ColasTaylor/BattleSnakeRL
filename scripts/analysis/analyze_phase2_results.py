# -*- coding: utf-8 -*-
"""
Phase 2 analysis for PPO vs random heuristics and PPO+HMM.
Outputs:
  - Training curves from logged CSVs
  - Evaluation curves (short eval runs with random heuristic opponents, seeds=[0,1,2])
  - Comparison bar chart of mean eval return
"""

import argparse
import random
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
from snake_rl.opponent_model.hmm import HMMOpponentModel  # noqa: E402

BASE_DIR = PROJECT_ROOT

# Paths (adjust if needed)
PPO_TRAIN_CSV = BASE_DIR / "logs" / "ppo_randomheuristic_train.csv"
PPO_HMM_TRAIN_CSV = BASE_DIR / "logs_ppo_hmm" / "train_ppo_hmm.csv"
PPO_MODEL = BASE_DIR / "models" / "ppo_randomheuristic.pt"
PPO_HMM_MODEL = BASE_DIR / "models" / "ppo_hmm_two_snake.pt"
SINGLE_SNAKE_MODEL = BASE_DIR / "checkpoints" / "Singlesnake_ppo_snake.pt"

PLOT_DIR = BASE_DIR / "phase2_plots"
PLOT_DIR.mkdir(exist_ok=True)

SEEDS = list(range(10))  # 10 seeds
EPISODES_PER_SEED = 20  # 20 episodes per seed


def sample_opponent(num_actions: int) -> Tuple[object, str]:
    """Pick a random heuristic opponent and return (agent, name)."""
    options = [
        ("random", RandomAgent),
        ("greedy_food", GreedyFoodAgent),
        ("heuristic_safe", SafeSpaceHeuristicAgent),
        ("loopy_bot", LoopyBotAgent),
        ("right_bot", RightBotAgent),
    ]
    name, ctor = random.choice(options)
    return ctor(num_actions=num_actions, name=name), name


def eval_ppo_vs_heuristics(
    model_path: Path,
    grid_size: int = 15,
    seeds: Sequence[int] = SEEDS,
    episodes_per_seed: int = EPISODES_PER_SEED,
) -> List[float]:
    """Evaluate PPO against heuristic opponents and return episodic returns."""
    returns: List[float] = []
    env = SnakeGridEnv(
        grid_size=grid_size, num_snakes=2, render_mode="none", max_steps=500
    )
    obs_shape = env.reset()[0].shape
    agent = PPOAgent(obs_shape=obs_shape, num_actions=4)
    agent.load(str(model_path))

    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)
        for _ in range(episodes_per_seed):
            obs_n = env.reset()
            opponent, _ = sample_opponent(num_actions=4)
            done = False
            ep_ret = 0.0
            while not done:
                a0 = agent.act(obs_n[0], deterministic=True)
                a1 = opponent.act(obs_n[1], deterministic=True)
                obs_n, rew_n, done, _ = env.step([a0, a1])
                ep_ret += rew_n[0]
            returns.append(ep_ret)
    env.close()
    return returns


def eval_ppo_hmm_vs_heuristics(
    model_path: Path,
    grid_size: int = 11,
    seeds: Sequence[int] = SEEDS,
    episodes_per_seed: int = EPISODES_PER_SEED,
) -> List[float]:
    """Evaluate PPO+HMM against heuristic opponents and return episodic returns."""
    returns: List[float] = []
    env = SnakeGridEnv(
        grid_size=grid_size, num_snakes=2, render_mode="none", max_steps=500
    )
    hmm = HMMOpponentModel()

    obs_n = env.reset()
    hmm.reset(env, learning_idx=0, opponent_idx=1)
    base_obs = obs_n[0].astype(np.float32).flatten()
    belief = hmm.get_belief().astype(np.float32)
    obs_aug = np.concatenate([base_obs, belief], axis=0)
    obs_shape = obs_aug.shape

    agent = PPOAgent(obs_shape=obs_shape, num_actions=4)
    agent.load(str(model_path))

    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)
        for _ in range(episodes_per_seed):
            obs_n = env.reset()
            hmm.reset(env, learning_idx=0, opponent_idx=1)
            base_obs = obs_n[0].astype(np.float32).flatten()
            belief = hmm.get_belief().astype(np.float32)
            obs_aug = np.concatenate([base_obs, belief], axis=0)

            opponent, _ = sample_opponent(num_actions=4)
            done = False
            ep_ret = 0.0
            while not done:
                a0 = agent.act(obs_aug, deterministic=True)
                a1 = opponent.act(obs_n[1], deterministic=True)
                obs_n, rew_n, done, _ = env.step([a0, a1])
                hmm.update_from_env(env)

                base_obs = obs_n[0].astype(np.float32).flatten()
                belief = hmm.get_belief().astype(np.float32)
                obs_aug = np.concatenate([base_obs, belief], axis=0)

                ep_ret += rew_n[0]
            returns.append(ep_ret)
    env.close()
    return returns


def plot_training_curves() -> None:
    """Plot training curves from logged CSVs, if available."""
    if PPO_TRAIN_CSV.exists():
        ppo_train = pd.read_csv(PPO_TRAIN_CSV)
        plt.figure(figsize=(7, 4))
        plt.plot(
            ppo_train["total_steps"],
            ppo_train["ma_reward"],
            label="PPO random heuristics",
        )
        plt.xlabel("Env steps")
        plt.ylabel("MA return")
        plt.title("Training Curve - PPO vs random heuristics")
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "phase2_training_ppo.png", dpi=300)
    if PPO_HMM_TRAIN_CSV.exists():
        hmm_train = pd.read_csv(PPO_HMM_TRAIN_CSV)
        plt.figure(figsize=(7, 4))
        plt.plot(
            hmm_train["steps"],
            hmm_train["return"].rolling(20, min_periods=1).mean(),
            label="PPO+HMM",
        )
        plt.xlabel("Env steps")
        plt.ylabel("Return")
        plt.title("Training Curve - PPO+HMM vs random heuristics")
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "phase2_training_ppohmm.png", dpi=300)


def plot_eval_curves(ppo_returns: List[float], hmm_returns: List[float]) -> None:
    """Plot rolling evaluation curves for PPO and PPO+HMM."""
    plt.figure(figsize=(7, 4))
    plt.plot(
        np.arange(len(ppo_returns)),
        pd.Series(ppo_returns).rolling(5, min_periods=1).mean(),
        label="PPO eval",
    )
    plt.plot(
        np.arange(len(hmm_returns)),
        pd.Series(hmm_returns).rolling(5, min_periods=1).mean(),
        label="PPO+HMM eval",
    )
    plt.xlabel("Eval episodes")
    plt.ylabel("Mean return (rolling)")
    plt.title("Evaluation curves vs random heuristics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "phase2_eval_curves.png", dpi=300)


def plot_comparison_bar(ppo_returns: List[float], hmm_returns: List[float]) -> None:
    """Plot mean/variance comparison between PPO variants."""
    means = [np.mean(ppo_returns), np.mean(hmm_returns)]
    stds = [np.std(ppo_returns), np.std(hmm_returns)]
    labels = ["PPO", "PPO+HMM"]
    plt.figure(figsize=(5, 4))
    plt.bar(labels, means, yerr=stds, capsize=6)
    plt.ylabel("Mean eval return")
    plt.title("PPO vs PPO+HMM (random heuristic opponents)")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "phase2_comparison_bar.png", dpi=300)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2 analysis: PPO vs random heuristics, PPO+HMM, and optional single-snake baseline"
    )
    parser.add_argument(
        "--single_snake_eval",
        action="store_true",
        help="Also evaluate the single-snake PPO baseline vs random heuristics",
    )
    args = parser.parse_args()

    plot_training_curves()

    print("Running eval for PPO vs random heuristics...")
    ppo_ret = eval_ppo_vs_heuristics(PPO_MODEL)
    print(f"PPO eval mean return: {np.mean(ppo_ret):.3f}")

    print("Running eval for PPO+HMM vs random heuristics...")
    hmm_ret = eval_ppo_hmm_vs_heuristics(PPO_HMM_MODEL)
    print(f"PPO+HMM eval mean return: {np.mean(hmm_ret):.3f}")

    # Optional: evaluate single-snake PPO (trained in 1-snake env) in 2-snake vs heuristics
    single_ret = None
    if args.single_snake_eval and SINGLE_SNAKE_MODEL.exists():
        print("Running eval for single-snake PPO baseline vs random heuristics...")
        single_ret = eval_ppo_vs_heuristics(SINGLE_SNAKE_MODEL)
        print(f"Single-snake PPO eval mean return: {np.mean(single_ret):.3f}")

    plot_eval_curves(ppo_ret, hmm_ret)
    plot_comparison_bar(ppo_ret, hmm_ret)
    if single_ret is not None:
        # Append to comparison with error bars
        plt.figure(figsize=(6, 4))
        labels = ["PPO", "PPO+HMM", "Single PPO"]
        means = [np.mean(ppo_ret), np.mean(hmm_ret), np.mean(single_ret)]
        stds = [np.std(ppo_ret), np.std(hmm_ret), np.std(single_ret)]
        plt.bar(labels, means, yerr=stds, capsize=6)
        plt.ylabel("Mean eval return")
        plt.title("PPO family vs random heuristic opponents")
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "phase2_comparison_bar_with_single.png", dpi=300)

    print(f"Plots saved to {PLOT_DIR}")


if __name__ == "__main__":
    main()

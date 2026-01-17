# train_ppo_hmm_two_snake.py

import argparse
import csv
import os
import random
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

from snake_rl.agents.heuristic_agents import (  # noqa: E402
    GreedyFoodAgent,
    SafeSpaceHeuristicAgent,
    LoopyBotAgent,
    RightBotAgent,
)
from snake_rl.agents.ppo_agent import PPOAgent  # noqa: E402
from snake_rl.envs.snake_env import SnakeGridEnv  # noqa: E402
from snake_rl.opponent_model.hmm_opponent_model import HMMOpponentModel  # noqa: E402


class RandomOpponent:
    """Very simple opponent: picks random valid action 0..3."""

    def __init__(self, num_actions: int = 4) -> None:
        self.num_actions = num_actions

    def act(self, obs: np.ndarray) -> int:
        return np.random.randint(self.num_actions)


def sample_opponent(num_actions: int) -> Tuple[object, str]:
    """Pick a random heuristic opponent each episode."""
    options = [
        ("random", RandomOpponent),
        ("hungry", GreedyFoodAgent),
        ("scared", SafeSpaceHeuristicAgent),
        ("loopy", LoopyBotAgent),
        ("right", RightBotAgent),
    ]
    name, ctor = random.choice(options)
    return ctor(num_actions=num_actions), name


def make_env_and_hmm(
    grid_size: int = 11,
    max_steps: int = 500,
) -> Tuple[SnakeGridEnv, HMMOpponentModel, int]:
    """Create a 2-snake env and an HMMOpponentModel, and return:

    - env
    - hmm
    - augmented obs dimension for learning snake
    """
    env = SnakeGridEnv(
        grid_size=grid_size,
        num_snakes=2,
        max_steps=max_steps,
        render_mode="none",
    )

    hmm = HMMOpponentModel()

    # Do a dummy reset to figure out obs_dim
    obs_n = env.reset()
    hmm.reset(env, learning_idx=0, opponent_idx=1)

    base_obs = obs_n[0].astype(np.float32).flatten()
    belief = hmm.get_belief().astype(np.float32)
    aug_obs = np.concatenate([base_obs, belief], axis=0)
    obs_dim = aug_obs.shape[0]

    return env, hmm, obs_dim


def evaluate_ppo_hmm(
    env: SnakeGridEnv,
    hmm: HMMOpponentModel,
    agent: PPOAgent,
    opponent,
    num_episodes: int = 20,
    max_steps: int = 500,
) -> Dict[str, float]:
    """Run evaluation with PPO (snake 0) vs opponent (snake 1),
    using HMM-augmented observations.
    """
    ep_returns = []
    ep_food = []

    for _ in range(num_episodes):
        obs_n = env.reset()
        hmm.reset(env, learning_idx=0, opponent_idx=1)

        # Build initial augmented obs for snake 0
        base_obs = obs_n[0].astype(np.float32).flatten()
        belief = hmm.get_belief().astype(np.float32)
        obs_aug = np.concatenate([base_obs, belief], axis=0)

        done = False
        total_r = 0.0
        steps = 0

        while not done and steps < max_steps:
            steps += 1

            # Actions
            a0 = agent.act(obs_aug, deterministic=True)
            opp_obs = obs_n[1]  # opponent sees normal grid
            a1 = opponent.act(opp_obs)

            obs_n, rew_n, done, info = env.step([a0, a1])
            hmm.update_from_env(env)

            # Next augmented obs
            base_obs = obs_n[0].astype(np.float32).flatten()
            belief = hmm.get_belief().astype(np.float32)
            obs_aug = np.concatenate([base_obs, belief], axis=0)

            total_r += rew_n[0]

        ep_returns.append(total_r)
        ep_food.append(env.snakes[0].score)

    return {
        "eval_mean_return": float(np.mean(ep_returns)),
        "eval_std_return": float(np.std(ep_returns)),
        "eval_mean_food": float(np.mean(ep_food)),
        "eval_std_food": float(np.std(ep_food)),
    }


def train_ppo_hmm_two_snake(
    total_steps: int = 300_000,
    grid_size: int = 11,
    max_steps_per_ep: int = 500,
    log_dir: str = "logs_ppo_hmm",
    save_path: str = "models/ppo_hmm_two_snake.pt",
    lr: float = 3e-4,
    entropy_coef: float = 0.0,
    clip_range: float = 0.3,
    gae_lambda: float = 0.95,
    num_epochs: int = 4,
    init_checkpoint: Optional[str] = None,
) -> Tuple[PPOAgent, SnakeGridEnv, HMMOpponentModel]:
    """Train PPO with HMM-augmented observations in a two-snake setting."""
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    env, hmm, obs_dim = make_env_and_hmm(
        grid_size=grid_size, max_steps=max_steps_per_ep
    )

    obs_shape = (obs_dim,)
    num_actions = 4

    agent = PPOAgent(
        obs_shape=obs_shape,
        num_actions=num_actions,
        name="ppo_hmm_two_snake",
        lr=lr,
        entropy_coef=entropy_coef,
        clip_eps=clip_range,
        lam=gae_lambda,
        num_epochs=num_epochs,
    )
    if init_checkpoint:
        try:
            agent.load(init_checkpoint)
            print(f"[INIT] Loaded PPO weights from {init_checkpoint}")
        except Exception as e:
            print(f"[WARN] Could not load init checkpoint {init_checkpoint}: {e}")

    step_count = 0
    episode = 0
    recent_returns: Deque[float] = deque(maxlen=50)

    train_log_path = os.path.join(log_dir, "train_ppo_hmm.csv")
    eval_log_path = os.path.join(log_dir, "eval_ppo_hmm.csv")

    train_f = open(train_log_path, "w", newline="")
    eval_f = open(eval_log_path, "w", newline="")
    train_writer = csv.writer(train_f)
    eval_writer = csv.writer(eval_f)

    train_writer.writerow(["episode", "steps", "return", "len"])
    eval_writer.writerow(
        [
            "episode",
            "steps",
            "eval_mean_return",
            "eval_std_return",
            "eval_mean_food",
            "eval_std_food",
        ]
    )

    best_eval_food = float("-inf")

    try:
        while step_count < total_steps:
            # -------- One training episode --------
            opponent, opponent_kind = sample_opponent(num_actions=num_actions)

            obs_n = env.reset()
            hmm.reset(env, learning_idx=0, opponent_idx=1)

            base_obs = obs_n[0].astype(np.float32).flatten()
            belief = hmm.get_belief().astype(np.float32)
            obs_aug = np.concatenate([base_obs, belief], axis=0)

            done = False
            ep_return = 0.0
            ep_len = 0

            while not done and ep_len < max_steps_per_ep and step_count < total_steps:
                ep_len += 1
                step_count += 1

                # PPO action
                a0 = agent.act(obs_aug, deterministic=False)
                # Opponent
                opp_obs = obs_n[1]
                a1 = opponent.act(opp_obs)

                next_obs_n, rew_n, done, info = env.step([a0, a1])
                hmm.update_from_env(env)

                # Build next augmented obs for snake 0
                base_next = next_obs_n[0].astype(np.float32).flatten()
                belief_next = hmm.get_belief().astype(np.float32)
                next_obs_aug = np.concatenate([base_next, belief_next], axis=0)

                r0 = rew_n[0]
                ep_return += r0

                # Store transition for PPO (adapt keys to your PPOAgent API)
                agent.observe(
                    {
                        "obs": obs_aug,
                        "action": a0,
                        "reward": r0,
                        "next_obs": next_obs_aug,
                        "done": done,
                    }
                )

                obs_n = next_obs_n
                obs_aug = next_obs_aug

                # Trigger PPO update (depends on your impl;
                # if you have a rollout mechanism, replace this).
                _ = agent.update()

            episode += 1
            recent_returns.append(ep_return)
            train_writer.writerow([episode, step_count, ep_return, ep_len])

            print(
                f"[TRAIN-HMM] Ep={episode:4d} | steps={step_count:7d} | "
                f"return={ep_return:7.2f} | MA(50)={np.mean(recent_returns):7.2f} | opp={opponent_kind}"
            )

            # -------- Periodic evaluation --------
            if episode % 20 == 0:
                eval_stats = evaluate_ppo_hmm(
                    env,
                    hmm,
                    agent,
                    opponent,
                    num_episodes=20,
                    max_steps=max_steps_per_ep,
                )
                print(
                    f"[EVAL-HMM] Ep={episode:4d} | steps={step_count:7d} | "
                    f"mean_return={eval_stats['eval_mean_return']:6.2f} | "
                    f"mean_food={eval_stats['eval_mean_food']:5.2f}"
                )

                eval_writer.writerow(
                    [
                        episode,
                        step_count,
                        eval_stats["eval_mean_return"],
                        eval_stats["eval_std_return"],
                        eval_stats["eval_mean_food"],
                        eval_stats["eval_std_food"],
                    ]
                )

                # Save best model by mean food eaten
                if eval_stats["eval_mean_food"] > best_eval_food:
                    best_eval_food = eval_stats["eval_mean_food"]
                    print(
                        f"[BEST-HMM] New best mean_food={best_eval_food:.2f}, saving model..."
                    )
                    agent.save(save_path)

    finally:
        train_f.close()
        eval_f.close()

    return agent, env, hmm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--total_steps", type=int, default=300_000)
    parser.add_argument("--grid_size", type=int, default=11)
    parser.add_argument("--max_steps_per_ep", type=int, default=500)
    parser.add_argument("--log_dir", type=str, default="logs_ppo_hmm")
    parser.add_argument("--save_path", type=str, default="models/ppo_hmm_two_snake.pt")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--entropy_coef", type=float, default=0.0)
    parser.add_argument("--clip_range", type=float, default=0.3)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--num_epochs", type=int, default=4)
    parser.add_argument("--init_checkpoint", type=str, default=None)
    args = parser.parse_args()

    start = time.time()
    agent, env, hmm = train_ppo_hmm_two_snake(
        total_steps=args.total_steps,
        grid_size=args.grid_size,
        max_steps_per_ep=args.max_steps_per_ep,
        log_dir=args.log_dir,
        save_path=args.save_path,
        lr=args.lr,
        entropy_coef=args.entropy_coef,
        clip_range=args.clip_range,
        gae_lambda=args.gae_lambda,
        num_epochs=args.num_epochs,
        init_checkpoint=args.init_checkpoint,
    )
    print("Training finished in {:.1f} seconds".format(time.time() - start))

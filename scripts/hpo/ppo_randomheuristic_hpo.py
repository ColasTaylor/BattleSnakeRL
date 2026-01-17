import itertools
import math
import random
import subprocess
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import pandas as pd

# Simple random search launcher for PPO vs random heuristics.
# Runs short 50k-step trainings with sampled hyperparameters and picks the best average return.

SCRIPT_DIR = Path(__file__).parent
TRAIN_SCRIPT = SCRIPT_DIR.parent / "train" / "train_ppo_randomheuristic_snake.py"
OUTPUT_DIR = SCRIPT_DIR / "hpo_runs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Search space
LR_LIST = [1e-4, 3e-4, 1e-3]
ENTROPY_LIST = [0.0, 0.01, 0.02]
CLIP_LIST = [0.1, 0.2, 0.3]
GAE_LIST = [0.9, 0.95, 0.99]
NUM_EPOCHS_LIST = [3, 4]

TOTAL_STEPS = 50_000
N_TRIALS = 8


def sample_configs(n: int) -> List[Tuple[float, float, float, float, int]]:
    all_configs = list(
        itertools.product(LR_LIST, ENTROPY_LIST, CLIP_LIST, GAE_LIST, NUM_EPOCHS_LIST)
    )
    random.shuffle(all_configs)
    return all_configs[:n]


def run_trial(idx: int, cfg: Sequence[float]) -> dict:
    lr, ent, clip, gae_lam, num_epochs = cfg
    run_name = f"trial_{idx:02d}_lr{lr}_ent{ent}_clip{clip}_lam{gae_lam}_ep{num_epochs}"
    log_path = OUTPUT_DIR / f"{run_name}.csv"
    save_path = OUTPUT_DIR / f"{run_name}.pt"

    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--total_steps",
        str(TOTAL_STEPS),
        "--log_path",
        str(log_path),
        "--save_path",
        str(save_path),
        "--lr",
        str(lr),
        "--entropy_coef",
        str(ent),
        "--clip_range",
        str(clip),
        "--gae_lambda",
        str(gae_lam),
        "--num_epochs",
        str(num_epochs),
    ]

    print(f"[HPO] Running {run_name} ...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(
            f"[HPO][FAIL] {run_name}: returncode={result.returncode}\n{result.stderr}"
        )
        return {
            "run": run_name,
            "lr": lr,
            "entropy": ent,
            "clip": clip,
            "gae_lambda": gae_lam,
            "num_epochs": num_epochs,
            "score": -math.inf,
            "log_path": str(log_path),
            "error": result.stderr,
        }

    score = compute_score(log_path)
    print(f"[HPO][DONE] {run_name} score={score:.3f}")
    return {
        "run": run_name,
        "lr": lr,
        "entropy": ent,
        "clip": clip,
        "gae_lambda": gae_lam,
        "num_epochs": num_epochs,
        "score": score,
        "log_path": str(log_path),
        "error": None,
    }


def compute_score(log_path: Path, tail: int = 20) -> float:
    if not log_path.exists():
        return -math.inf
    df = pd.read_csv(log_path)
    if "ep_reward" not in df.columns:
        return -math.inf
    tail_df = df.tail(tail)
    return float(tail_df["ep_reward"].mean())


def main() -> None:
    configs = sample_configs(N_TRIALS)
    results = []
    for i, cfg in enumerate(configs):
        results.append(run_trial(i, cfg))

    results_df = pd.DataFrame(results)
    results_csv = OUTPUT_DIR / "hpo_results.csv"
    results_df.to_csv(results_csv, index=False)
    print("\n[HPO] Results written to", results_csv)

    best = results_df.sort_values("score", ascending=False).iloc[0]
    print("\n[HPO] Best config:")
    print(best.to_string())


if __name__ == "__main__":
    main()

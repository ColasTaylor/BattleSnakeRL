# BattleSnakeRL

Train agents, watch them slither, and occasionally crash into walls in the name of
science. This is a compact Snake + RL playground with DQN/PPO agents, heuristics,
and a handful of scripts to train, evaluate, and battle.

## Quick Start

Python 3.x is assumed. Create your environment, then:

```bash
# Run a simple turtle-based demo
python demos/turtle_snake_demo.py

# Watch a trained (or heuristic) agent play
python demos/watch_snake.py heuristic

# Train agents
python scripts/train/train_dqn.py
python scripts/train/train_ppo.py

# Run tests
python -m pytest
```

## Project Layout

```
src/snake_rl/          Core library (envs, agents, opponent model, eval)
scripts/train/         Training entry points
scripts/analysis/      Analysis utilities
scripts/hpo/           HPO scripts (optional)
demos/                 Visual demos / watchers
tests/                 Pytest-based tests
```

## Agents & Environment

- Environment: `snake_grid_env.py` (grid-based snake with reset/step/obs)
- Agents: `dqn.py`, `ppo.py`, `heuristics.py`
- Opponent model: `hmm.py`
- Evaluation: `eval/battle.py` (run matches, compute metrics)

## Notes

- Torch is required for DQN/PPO agents; heuristic agents and demos can run without it.
- Checkpoints are typically stored in `checkpoints/` (create as needed).

## Why This Exists

To explore multi-agent snake RL with a simple, hackable codebase. Bonus points if
you can make the snake stop chasing its own tail.

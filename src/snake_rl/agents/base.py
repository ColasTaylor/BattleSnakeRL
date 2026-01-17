from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np


class Agent(ABC):
    """Abstract base class for all agents in this project."""

    def __init__(self, num_actions: int, name: str = "agent") -> None:
        self.num_actions = num_actions
        self.name = name

    @abstractmethod
    def act(self, obs: np.ndarray, deterministic: bool = False) -> int:
        """Given a single observation, return an action index [0, num_actions)."""
        raise NotImplementedError

    def observe(self, transition: Dict[str, Any]) -> None:
        """Optionally store a step of experience. Default: do nothing."""
        return None

    def update(self) -> Dict[str, Any]:
        """Optionally run a learning update. Return dict of stats."""
        return {}

    def save(self, path: str) -> None:
        """Optionally save parameters to disk."""
        return None

    def load(self, path: str) -> None:
        """Optionally load parameters from disk."""
        return None

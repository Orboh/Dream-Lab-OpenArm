from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Robot(ABC):
    """Minimal bimanual robot interface, in canonical joint units (see orboh.types)."""

    name = "robot"

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @property
    @abstractmethod
    def is_connected(self) -> bool: ...

    @abstractmethod
    def read_joints(self) -> np.ndarray:
        """Measured joint positions, shape (16,)."""

    @abstractmethod
    def send_joints(self, q: np.ndarray) -> None:
        """Command target joint positions, shape (16,). Must return quickly."""

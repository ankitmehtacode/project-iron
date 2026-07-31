from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict
import numpy as np


class ModelWrapper(ABC):
    def __init__(self, model_path: str, device: str = "CPU"):
        self.model_path = Path(model_path)
        self.device = device
        self.model = None
        self._validate_path()

    def _validate_path(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

    @abstractmethod
    def load(self) -> None:
        """Load model into memory."""
        pass

    @abstractmethod
    def predict(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Run inference on inputs."""
        pass

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(path={self.model_path}, device={self.device})"
        )

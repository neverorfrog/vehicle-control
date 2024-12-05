from abc import ABC, abstractmethod

import numpy as np


class Controller(ABC):
    """Controller Class"""

    def __init__(self, kp: np.ndarray, kd: np.ndarray):
        self.kp = kp
        self.kd = kd

    @abstractmethod
    def command(self, *args, **kwargs):
        """Compute the control actions
        Returns:
            (np.array): control actions
        """
        pass

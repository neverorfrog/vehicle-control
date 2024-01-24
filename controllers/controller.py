import numpy as np
from typing import Tuple

class Controller():
    def __init__(self, kp: np.ndarray, kd: np.ndarray):
        self.kp = kp
        self.kd = kd
            
    def command(self, q_k, qd_k, ref_k):
        """Compute the control actions
        Args:
            q_k (np.array): current state
            qd_k (np.array): current derivative of the state
            ref_k (dict)
        Returns:
            (np.array): control actions
        """
        return np.array([1.0,0])
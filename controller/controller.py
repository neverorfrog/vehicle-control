import numpy as np
from typing import Tuple
from abc import abstractmethod
from abc import ABC

class Controller(ABC):
    '''Controller Class'''
    def __init__(self, kp: np.ndarray, kd: np.ndarray):
        self.kp = kp
        self.kd = kd
     
    @abstractmethod       
    def command(self, s_k):
        """Compute the control actions
        Args:
            q_k (np.array): current state
            qd_k (np.array): current derivative of the state
            ref_k (dict)
        Returns:
            (np.array): control actions
        """
        pass
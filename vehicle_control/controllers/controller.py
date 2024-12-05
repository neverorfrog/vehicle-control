import numpy as np
from abc import abstractmethod
from abc import ABC

class Controller(ABC):
    '''Controller Class'''
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
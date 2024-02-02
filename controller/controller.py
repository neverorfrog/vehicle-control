import numpy as np
from abc import abstractmethod
from abc import ABC
from environment.trajectory import Trajectory
from model.robot import Robot

class Controller(ABC):
    '''Controller Class'''
    def __init__(self, kp: np.ndarray, kd: np.ndarray):
        self.kp = kp
        self.kd = kd
     
    @abstractmethod       
    def command(self, robot: Robot, reference: Trajectory = None):
        """Compute the control actions
        Args:
            robot: robot that encapsulates current state and last applied action
            reference:  (optional) for feedback control
        Returns:
            (np.array): control actions
        """
        pass
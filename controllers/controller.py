import numpy as np
from typing import Tuple

class Controller():
    def command(self, q_k, qd_k, ref_k) -> Tuple[np.ndarray, bool]:
        """Compute the control actions
        Args:
            q_k (np.array): current state
            qd_k (np.array): current derivative of the state
            ref_k (dict): current reference 
        Returns:
            (np.array, bool): control actions and terminated condition
        """
        return np.array([1.0,0]), False
            
    def set_gains(self, kp = None, kd = None):
        if kp is not None:
            self.kp = kp
        if kd is not None:
            self.kd = kd
            
    def check_termination(self, e, ed):
        e = np.abs(e)
        ed = np.abs(ed)
        position_ok = all(e < self.threshold) == True
        velocity_ok = all(ed < self.threshold) == True
        time_ok = self.t[-1] >= self.T
        return time_ok and position_ok and velocity_ok
import numpy as np
from typing import Tuple

class Controller():
    def command(self, q_k, qd_k, t_k, reference) -> Tuple[np.ndarray, bool]:
        '''
        Given the reference and current state
        Outputs the control action given a certain control law
        '''
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
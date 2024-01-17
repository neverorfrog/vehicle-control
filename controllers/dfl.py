import sys
sys.path.append("..")

from simulation.model import *
from controllers.control import Control
from typing import Tuple
import numpy as np
from controllers.trajectory import *

class DFL(Control):
    def __init__(self, model: Model, dt):
        super().__init__(model, dt) 
        self.v_k = 0.001 #initial condition for integrator     
                 
    def command(self, q_k, qd_k, t_k, reference: Trajectory) -> Tuple[np.ndarray, bool]:
        
        ref_k = reference.update(t_k)
        
        e_p = ref_k['p'] - q_k[:-1]
        e_d = ref_k['pd'] - qd_k[:-1]
        u_k = ref_k['pdd'] + e_p*self.kp + e_d*self.kd
        
        theta = q_k[2]
        
        inverse_decoupling_matrix = np.array([
            [np.cos(theta), np.sin(theta)],
            [-np.sin(theta)/self.v_k, np.cos(theta)/self.v_k]])
        
        a_w = np.matmul(inverse_decoupling_matrix,u_k)
        self.v_k = self.v_k + self.dt * a_w[0]
        
        return np.array([self.v_k, a_w[1]]), False
        
import sys
sys.path.append("..")

from modeling.robot import *
from typing import Tuple
import numpy as np
from controllers.controller import Controller

class DFL(Controller):
    def __init__(self):
        self.v_k = 0.01 #initial condition for integrator 
        self.dt = 0.01 #for integration from acceleration to velocity    
                 
    def command(self, q_k, qd_k, t_k, reference) -> Tuple[np.ndarray, bool]:
        
        ref_k = reference.update(t_k)
        
        e_p = ref_k['p'] - q_k[:2]
        e_d = ref_k['pd'] - qd_k[:2]
        u_k = ref_k['pdd'] + e_p*self.kp + e_d*self.kd
        
        theta = q_k[2]
        
        inverse_decoupling_matrix = np.array([
            [np.cos(theta), np.sin(theta)],
            [-np.sin(theta)/self.v_k, np.cos(theta)/self.v_k]])
        
        a_w = np.matmul(inverse_decoupling_matrix,u_k)
        self.v_k = self.v_k + self.dt * a_w[0]
        
        return np.array([self.v_k, a_w[1]]), False
        
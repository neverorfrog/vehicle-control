import sys
sys.path.append("..")

from modeling.robot import *
from typing import Tuple
import numpy as np
from controllers.controller import Controller

class FBL(Controller):
    def __init__(self, b = 0.1):
        self.b = b
        
    def command(self, q_k, qd_k, ref_k) -> Tuple[np.ndarray, bool]:
        # TODO hardcodato
        x = q_k[0]
        y = q_k[1]
        theta = q_k[2]
        
        # point at distance b from center
        x_b = x + self.b * np.cos(theta)
        y_b = y + self.b * np.sin(theta)
        
        e_p = ref_k['p'] - [x_b,y_b]
        
        u_io = ref_k['pd'] + self.kp*e_p
        
        inverse_decoupling_matrix = np.array([
            [np.cos(theta), np.sin(theta)],
            [-np.sin(theta)/self.b, np.cos(theta)/self.b]])

        vw = np.matmul(inverse_decoupling_matrix,u_io)
    
        return vw, False

# TODO acceleration input
class DFBL(Controller):
    def __init__(self):
        self.v_k = 1 #initial condition for integrator 
        self.dt = 0.01 #for integration from acceleration to velocity    
                 
    def command(self, q_k, qd_k, ref_k) -> Tuple[np.ndarray, bool]:
        
        # TODO hardcodato
        x = q_k[0]
        y = q_k[1]
        theta = q_k[2]
        xd = qd_k[0]
        yd = qd_k[1]
        
        e_p = ref_k['p'] - [x,y]
        e_d = ref_k['pd'] - [xd,yd]
        u_io = ref_k['pdd'] + e_p*self.kp + e_d*self.kd
                
        inverse_decoupling_matrix = np.array([
            [np.cos(theta), np.sin(theta)],
            [-np.sin(theta)/self.v_k, np.cos(theta)/self.v_k]])
        
        a_w = np.matmul(inverse_decoupling_matrix,u_io)
        self.v_k = self.v_k + self.dt * a_w[0]
        
        return np.array([self.v_k, a_w[1]]), False
        
import sys
sys.path.append("..")

from modeling.robot import *
from typing import Tuple
import numpy as np
from controllers.controller import Controller
from modeling.trajectory import Trajectory
from numpy import sin,cos,tan

class FBL(Controller):
    def __init__(self, kp: np.ndarray, kd: np.ndarray, b = 0.1):
        super().__init__(kp, kd)
        self.b = b
        
    def command(self, q_k, qd_k, ref_k):
        # TODO hardcodato
        x = q_k[0]
        y = q_k[1]
        theta = q_k[2]
        
        # point at distance b from center
        x_b = x + self.b * cos(theta)
        y_b = y + self.b * sin(theta)
            
        # intermediate control signal
        e_p = ref_k['p'] - [x_b,y_b]
        u_io = ref_k['pd'] + self.kp*e_p
        
        # linearization
        inverse_decoupling_matrix = np.array([
            [cos(theta), sin(theta)],
            [-sin(theta)/self.b, cos(theta)/self.b]])

        vw = np.matmul(inverse_decoupling_matrix,u_io)
    
        return vw
    
class BicycleFBL(Controller):
    def __init__(self, kp: np.ndarray, kd: np.ndarray, b = 0.1, l = 0.5):
        super().__init__(kp, kd)
        self.b = b
        self.l = l
        
    def command(self, q_k, qd_k, ref_k):
        # TODO hardcodato
        x = q_k[0]
        y = q_k[1]
        theta = q_k[2]
        phi = q_k[3]
                
        # point at distance b from center
        x_b = x + self.l * cos(theta) + self.b * cos(theta+phi)
        y_b = y + self.l * sin(theta) + self.b * sin(theta+phi)
        
        # intermediate control signal
        e_p = ref_k['p'] - [x_b,y_b]
        u_io = ref_k['pd'] + self.kp*e_p
        
        # linearization
        inverse_decoupling_matrix = np.linalg.inv(np.array([
            [cos(theta)-tan(phi)*(sin(theta)+self.b*sin(theta+phi)/self.l), -self.b*sin(theta+phi)],
            [sin(theta)+tan(phi)*(cos(theta)+self.b*cos(theta+phi)/self.l),  self.b*cos(theta+phi)]]))

        vw = np.matmul(inverse_decoupling_matrix,u_io)
    
        return vw

# TODO acceleration input works well that way?
class DFBL(Controller):
    def __init__(self, kp: np.ndarray, kd: np.ndarray):
        super().__init__(kp, kd)
        self.v_k = 0.05 #initial condition for integrator 
        self.dt = 0.01 #for integration from acceleration to velocity    
                 
    def command(self, q_k, qd_k, ref_k) -> Tuple[np.ndarray]:
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
            [cos(theta), -sin(theta)],
            [-sin(theta)/self.v_k, cos(theta)/self.v_k]])
        
        a_w = np.matmul(inverse_decoupling_matrix,u_io)
        self.v_k = self.v_k + self.dt * a_w[0]
        
        return np.array([self.v_k, a_w[1]])
        
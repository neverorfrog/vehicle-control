import sys
sys.path.append("..")

from simulation.model import *
from typing import Tuple
import numpy as np

class Control():   
    
    def __init__(self, model: Model, dt):
        self.model = model
        self.discrete_ode = ca.Function('discrete_ode', [model.q,model.u], [model.RK4(dt)])
        self.dt = dt
         
    def step(self, q_k, u_k):
        """
            - Given current (kth) q and u
            - Applies it for dt (given in Model construction)
            - Return numpy array of next q
        """
        next_q = self.discrete_ode(q_k,u_k).full().squeeze()
        next_qd = self.model.transition_function(next_q, u_k).full().squeeze()
        return next_q, next_qd
        
    def run(self, reference, threshold = 0.01, T = None) -> None: 
        # for offline plotting
        q_traj = [np.zeros((self.model.q_len))]
        qd_traj = [np.zeros((self.model.q_len))]
        u_traj = []
        time = [0]
        
        # control loop initialization
        arrived = False
        self.T = reference.T if T is None else T
        self.forced_termination = True if T is not None else False
        self.threshold = threshold
        q_k = q_traj[-1]
        qd_k = qd_traj[-1]
        # control loop
        while True:
            time.append(time[-1] + self.dt)
            if arrived or time[-1] >= T: break
            u_k, arrived = self.command(q_k, qd_k, time[-1], reference)
            q_k, qd_k = self.step(q_k,u_k)
            q_traj.append(q_k)
            qd_traj.append(qd_k)
            u_traj.append(u_k)
           
        return np.array(q_traj), np.array(u_traj)
         
    def command(self, q_k, qd_k, t_k, reference) -> Tuple[np.ndarray, bool]:
        '''
        Given the reference and current state
        Outputs the control action given a certain control law
        '''
        return np.ones((2)), False
            
    def set_gains(self, kp = None, kd = None):
        if kp is not None:
            self.kp = kp
        if kd is not None:
            self.kd = kd
            
    def set_dt(self, dt):
        self.dt = dt
        
    def check_termination(self, e, ed):
        e = np.abs(e)
        ed = np.abs(ed)
        position_ok = all(e < self.threshold) == True
        velocity_ok = all(ed < self.threshold) == True
        time_ok = self.t[-1] >= self.T
        if self.forced_termination: return time_ok
        return time_ok and position_ok and velocity_ok
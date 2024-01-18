import sys
sys.path.append("..")

from modeling.robot import *
from controllers.controller import Controller
import numpy as np

class Simulation():   
    def __init__(self, robot: Robot, controller: Controller, dt):
        self.robot = robot
        self.controller = controller
        self.discrete_ode = ca.Function('discrete_ode', [robot.q,robot.u], [robot.RK4(dt)])
        self.dt = dt
         
    def step(self, q_k, u_k):
        """
            - Given current (kth) q and u
            - Applies it for dt (given in Robot construction)
            - Return numpy array of next q
        """
        next_q = self.discrete_ode(q_k,u_k).full().squeeze()
        next_qd = self.robot.transition_function(next_q, u_k).full().squeeze()
        return next_q, next_qd
        
    def run(self, reference, threshold = 0.01, T = None, q0 : np.ndarray = None) -> None: 
        # for offline plotting
        q_traj = [np.zeros((self.robot.q_len))] if q0 is None else [q0]
        qd_traj = [np.zeros((self.robot.q_len))]
        u_traj = []
        time = [0]
        
        # control loop initialization
        arrived = False
        self.T = reference.T if T is None else T
        self.threshold = threshold
        q_k = q_traj[-1]
        qd_k = qd_traj[-1]
        # control loop
        while True:
            time.append(time[-1] + self.dt)
            if arrived or time[-1] >= T: break
            u_k, arrived = self.controller.command(q_k, qd_k, time[-1], reference)
            q_k, qd_k = self.step(q_k,u_k)
            q_traj.append(q_k)
            qd_traj.append(qd_k)
            u_traj.append(u_k)
           
        return np.array(q_traj), np.array(u_traj)
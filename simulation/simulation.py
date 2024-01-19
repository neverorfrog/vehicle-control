import sys

sys.path.append("..")

from modeling.robot import *
from controllers.controller import Controller
import numpy as np
from modeling.trajectory import Trajectory
from modeling.util import wrap

class Simulation():   
    def __init__(self, dt: float, robot: Robot, controller: Controller, reference: Trajectory, threshold = 0.01):
        self.robot = robot
        self.controller = controller
        self.reference = reference
        self.threshold = threshold
        self.discrete_ode = ca.Function('discrete_ode', [robot.q,robot.u], [robot.RK4(dt)])
        self.dt = dt
         
    def step(self, q_k, u_k):
        """
            - Given current (kth) q and u
            - Applies it for dt (given in Robot construction)
            - Return numpy array of next q
        """
        next_q = self.discrete_ode(q_k,u_k).full().squeeze()
        next_q[2] = wrap(next_q[2]) # TODO better solution to normalize angles?
        next_qd = self.robot.transition_function(next_q, u_k).full().squeeze()
        return next_q, next_qd
        
    def run(self, q0 : np.ndarray, qd0 : np.ndarray, T: float): 
        # for offline plotting
        q_traj = [q0]
        qd_traj = [qd0]
        ref_traj = [self.reference.update(0)['p']]
        u_traj = []
        time = [0]
        
        # control loop initialization
        q_k = q_traj[-1]
        qd_k = qd_traj[-1]
        # control loop
        while True:
            time.append(time[-1] + self.dt)
            if time[-1] >= T: break
            
            # getting reference
            ref_k = self.reference.update(time[-1])
            
            # applying control signal
            u_k = self.controller.command(q_k, qd_k, time[-1], self.reference)
            q_k, qd_k = self.step(q_k,u_k)
            
            # logging
            q_traj.append(q_k)
            qd_traj.append(qd_k)
            u_traj.append(u_k)
            ref_traj.append(ref_k['p'])
        
        return np.array(q_traj), np.array(u_traj), np.array(ref_traj)
    
    # TODO make use of this function (not just use time to check for termination)
    def check_termination(self, e, ed):
        position_ok = all(np.abs(e) < self.threshold) == True
        velocity_ok = all(np.abs(ed) < self.threshold) == True
        return position_ok and velocity_ok
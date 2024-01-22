import sys
sys.path.append("..")

from modeling.robot import *
from controllers.controller import Controller
import numpy as np
from modeling.track import Track
from modeling.trajectory import Trajectory
from modeling.util import wrap
from modeling.car import Car
from casadi import atan2

class RacingSimulation():   
    def __init__(self, dt: float, car: Car, controller: Controller):
        self.car = car
        self.controller = controller
        self.dt = dt
        self.pos = np.array([0,0]) # initial condition for plotting stuff
        self.transition = ca.Function('transition', [self.car.q, self.car.u, self.car.k], [self.car.integrate(self.dt)])
         
    def step(self, q_k: dict, u_k, curvature):
        """
            - Given current (kth) q and u
            - Applies it for dt (given in Robot construction)
            - Return numpy array of next q
        """
        q_k = np.array(list(q_k.values())) # going to integrate, so we need array
        next_q = self.transition(q_k,u_k,curvature).full().squeeze() # result of integration
        next_q = dict(zip(self.car.q_keys, next_q)) # everywhere else than integration dict is better
        curvature = self.car.track.get_curvature(next_q['s']) # curvature is not a continuous function of s
        return next_q, curvature
    
    def extract_pose(self, q):
        pos = self.car.track.get_global_position(q['s'],q['ey'])
        pos_dot = (pos - self.pos)/self.dt
        psi = atan2(pos_dot[1], pos_dot[0])
        self.pos = pos
        return np.array([self.pos[0], self.pos[1], psi, q['delta']])
        
    def run(self, q0 : dict, T: float): 
        # for offline plotting
        q_traj = [np.array([0,0,0,0])] # TODO hardcodato
        time = [0]
        u_traj = []
        
        # control loop
        q_k = q0
        curvature = 0 # TODO hardcodato
        while True:
            time.append(time[-1] + self.dt)
            if time[-1] >= T: break
            
            # applying control signal
            u_k = self.controller.command(np.array(list(q_k.values())), curvature) # returns a_k, w_k
            
            print(u_k)
            
            q_k, curvature = self.step(q_k,u_k,curvature)
            
            # logging
            q_traj.append(self.extract_pose(q_k))
            u_traj.append(u_k)
        
        return np.array(q_traj), np.array(u_traj)
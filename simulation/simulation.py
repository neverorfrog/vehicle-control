import sys
sys.path.append("..")

from modeling.robot import *
from controllers.controller import Controller
import numpy as np
from modeling.track import Track
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
        
    def run(self, q0 : dict, T: float):
         
        # for offline plotting
        q_traj = [self.car.extract_pose()] # extract pose without arguments gives all zeros
        u_traj = []
        
        # control loop
        q_k = q0
        curvature = 0 # TODO hardcodato
        while True:
            if q_k['t'] >= T: break
            
            # computing control signal
            u_k = self.controller.command(np.array(list(q_k.values())), curvature)
            
            # applying control signal
            q_k, curvature = self.step(q_k,u_k,curvature)
            
            # logging
            q_traj.append(self.car.extract_pose(q_k))
            u_traj.append(u_k)
        
        return np.array(q_traj), np.array(u_traj)
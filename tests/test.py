import sys
sys.path.append("..")

import numpy as np
from matplotlib import pyplot as plt
from modeling.track import Track
from modeling.bicycle import Bicycle
from simulation.simulation import RacingSimulation
from simulation.plotting import animate

import casadi as ca


if __name__ == "__main__":    
    wp = np.array([[-2,0],[2,0],[2,2],[-2,2],[-2,0],[0,0]])
    wp_x = wp[:,0]
    wp_y = wp[:,1]

    # Specify path resolution
    resolution = 0.05  # m / wp

    # Create reference path
    track = Track(wp_x, wp_y, resolution,smoothing=15, width=0.15)
    # track.show()
    
    # Bicycle model
    car = Bicycle(track, length=0.5, dt=0.05)
    wp = car.get_current_waypoint()
    
    dt = 0.01
    T = 1
    # Logging containers
    q_traj = [car.temporal_state]
    u_traj = []
    
    # control loop
    t = 0.0
    q_k = q_traj[0]
    while t < T:
        
        t += dt
        
        # computing control signal
        # u_k = self.controller.command(np.array(list(q_k.values())), curvature)
        
        # applying control signal
        # q_k, curvature = self.step(q_k,u_k,curvature)
        
        u_k = np.array([1,0,car.get_current_waypoint().kappa])
        q_k = car.drive(u_k)
        
        # logging
        q_traj.append(q_k)
        u_traj.append(u_k)
        
    animate(np.array(q_traj), np.array(u_traj), car, track)
    

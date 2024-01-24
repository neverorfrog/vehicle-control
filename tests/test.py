import sys
sys.path.append("..")

from modeling.state import TemporalState
import numpy as np
from matplotlib import pyplot as plt
from modeling.track import Track
from modeling.bicycle import Bicycle
from simulation.simulation import RacingSimulation
from simulation.plotting import animate
from controllers.new_new_mpc import RacingMPC

import casadi as ca


if __name__ == "__main__":    

    # Create reference path
    wp = np.array([[-2,0],[2,0],[2,2],[-2,2],[-2,0],[0,0]])
    track = Track(wp_x=wp[:,0], wp_y=wp[:,1], resolution=0.05,smoothing=15, width=0.15)
    
    # Bicycle model
    car = Bicycle(track, length=0.2, dt=0.05)
    car.temporal_state = TemporalState(x = 2, y = 2)

    # Logging containers
    s_traj = [car.temporal_state]
    i_traj = []
    
    # control loop
    dt = 0.01
    T = 1
    controller = RacingMPC(horizon = 10, dt = 0.01, car = car)
    t = 0.0
    s_k = s_traj[0]
    while t < T:
        t += dt
        
        # computing control signal
        i_k = controller.command(s_k)
        
        # applying control signal
        s_k = car.drive(i_k)
        
        # logging
        s_traj.append(s_k)
        i_traj.append(i_k)
        
    animate(np.array(s_traj), np.array(i_traj), car, track)
    

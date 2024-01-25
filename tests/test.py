import sys
import time
sys.path.append("..")

from modeling.state import RacingCarState
import numpy as np
from matplotlib import pyplot as plt
from modeling.track import Track
from modeling.racing_car import RacingCar
from simulation.simulation import RacingSimulation
from simulation.plotting import animate
from controllers.mpc import RacingMPC
import casadi as ca


if __name__ == "__main__":    

    # Create reference path
    wp = np.array([[-2,0],[2,0],[2,2],[-2,2],[-2,0],[0,0]])
    track = Track(wp_x=wp[:,0], wp_y=wp[:,1], resolution=0.05,smoothing=15, width=0.15)
    
    # Bicycle model
    car = RacingCar(track, length=0.2, dt=0.05)
    car.state = RacingCarState(x = 2, y = 4, psi = 3)
    
    # Logging containers
    s_traj = [car.state]
    i_traj = []
    
    # control loop
    dt = 0.01
    T = 5
    controller = RacingMPC(horizon = 5, dt = 0.01, car = car)
    t = 0.0
    s_k = s_traj[0]
    elapsed = []
    while t < T:
        t += dt
        
        # computing control signal
        start = time.time()
        i_k = controller.command(s_k)
        elapsed.append(time.time() - start)
        
        # applying control signal
        s_k = car.drive(i_k)
        
        # logging
        s_traj.append(s_k)
        i_traj.append(i_k)
    
    print(np.mean(elapsed))    
    animate(np.array(s_traj), np.array(i_traj), car, track)
    

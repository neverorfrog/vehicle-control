import sys
sys.path.append(".")

from modeling.state import RacingCarState
import numpy as np
from modeling.track import Track
from modeling.racing_car import RacingCar
from simulation.simulation import RacingSimulation
from simulation.plotting import animate
from controllers.mpc import RacingMPC

# Create reference path
wp = np.array([[-2,0],[2,0],[2,2],[-2,2],[-2,0],[0,0]])
track = Track(wp_x=wp[:,0], wp_y=wp[:,1], resolution=0.05,smoothing=15, width=0.15)

# Bicycle model
car = RacingCar(track, length=0.2, dt=0.05)
car.state = RacingCarState(x = 2, y = 4, psi = 3)

# MPC controller
controller = RacingMPC(horizon = 5, dt = 0.03, car = car)

# Simulation
simulation = RacingSimulation(car, controller)   
s_traj, i_traj = simulation.run(N = 50)
animate(np.array(s_traj), np.array(i_traj), car, track)   

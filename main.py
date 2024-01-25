import sys
sys.path.append(".")

from modeling.state import KinematicCarState
import numpy as np
from modeling.track import Track
from modeling.racing_car import KinematicCar
from simulation.simulation import RacingSimulation
from simulation.plotting import animate
from controllers.mpc import RacingMPC

# Create reference path
wp = np.array([[-2,0],[2,0],[2,2],[-2,2],[-2,0],[-0.5,0]])
track = Track(wp_x=wp[:,0], wp_y=wp[:,1], resolution=0.03,smoothing=25,width=0.4)

# Bicycle model
car = KinematicCar(track, length=0.2, dt=0.05)
car.state = KinematicCarState(x = -1.24812, y = 0.2, psi = np.pi/6, v=5)

# MPC controller
controller = RacingMPC(horizon = 3, dt = 0.03, car = car)

# Simulation
simulation = RacingSimulation(car, controller)   
s_traj, i_traj = simulation.run(N = 80)
# animate(np.array(s_traj), np.array(i_traj), car, track)   

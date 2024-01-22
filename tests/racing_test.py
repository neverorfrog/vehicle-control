import sys
sys.path.append("..")

from modeling.car import Car
from modeling.track import Track
from simulation.simulation import RacingSimulation
from controllers.mpc import RacingMPC
from simulation.plotting import animate
import numpy as np

track = Track()
car = Car(track)
controller = RacingMPC(horizon=5, dt = 0.03, car=car) 
simulation = RacingSimulation(dt=0.05, car=car, controller=controller) # every 50ms the MPC horizon is activated

# q_k : {'v', 'psi', 't', 'ey', 'epsi', 'delta', 's'}
q0=dict(zip(car.q_keys, np.array([5,0,0,0,0,0,5])))
q_traj, u_traj = simulation.run(q0, T=10)

animation = animate(q_traj, u_traj, car, track)
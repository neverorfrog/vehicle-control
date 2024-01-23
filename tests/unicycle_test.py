import sys
sys.path.append("..")

from simulation.simulation import *
from modeling import *
from modeling.track import *
from modeling.trajectory import *
from controllers.io_linearization import FBL
from simulation.plotting import animate
from modeling.unicycle import Unicycle
from controllers.unicycle_mpc import RacingMPC

track = Track()
car = Unicycle(track)
controller = RacingMPC(horizon=5, dt = 0.05, car=car) 
simulation = RacingSimulation(dt=0.1, car=car, controller=controller) # every 50ms the MPC horizon is activated
q0={'x': 0,'y': 0,'psi': 1,'s': 0,'ey': 0,'epsi': 1,'t': 0}
q_traj, u_traj = simulation.run(q0, T=2)
animation = animate(q_traj, u_traj, car, track)
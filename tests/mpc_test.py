import sys
sys.path.append("..")

from simulation.simulation import *
from modeling import *
from modeling.track import *
from modeling.trajectory import *
from controllers.mpc import *
from simulation.plotting import animate

robot = DifferentialDrive()
# robot = Bicycle()
track = Track(freq=0.02)
controller = MPC(horizon=30, dt=0.05, model=robot)
loop = Simulation(dt=0.05, robot=robot, controller=controller, reference=track)

q_traj, u_traj, ref_traj = loop.run(
    T = 50, 
    q0 = np.array([0,0,0]),
    qd0=np.array([0,0,0]))

animation = animate(q_traj, u_traj, ref_traj, robot, track)
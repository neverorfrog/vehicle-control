import sys
sys.path.append("..")

from simulation.simulation import *
from modeling import *
from modeling.track import *
from modeling.trajectory import *
from controllers.mpc import *
from simulation.plotting import animate

# robot = DifferentialDrive()
robot = Bicycle()
track = Track(freq=0.1)
controller = MPC(horizon=20, dt=0.05, model=robot)
loop = Simulation(dt=0.01, robot=robot, controller=controller, reference=track)

q_traj, u_traj, ref_traj = loop.run(
    T = 10, 
    q0 = np.array([0,0,0,0]),
    qd0=np.array([0,0,0,0]))

animation = animate(q_traj, u_traj, ref_traj, robot, track)
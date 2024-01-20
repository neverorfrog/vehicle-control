import sys
sys.path.append("..")

from simulation.simulation import *
from modeling.track import *
from modeling.bicycle import Bicycle
from modeling.trajectory import *
from controllers.io_linearization import BicycleFBL
from simulation.plotting import animate

track = Track(freq=0.03)
robot = Bicycle(track)
controller = BicycleFBL(kp=[1,1],kd=[1,1], b = 0.05, l = robot.l)

loop = Simulation(dt=0.1, robot=robot, controller=controller, reference=track)

q_traj, u_traj, ref_traj = loop.run(
    T = 50, 
    q0 = np.array([0,0,0,0,0,0,0,0]),
    qd0=np.array([0,0,0,0,0,0,0,0]))

animation = animate(q_traj, u_traj, ref_traj, robot, track)
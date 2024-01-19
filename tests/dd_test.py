import sys
sys.path.append("..")

from simulation.simulation import *
from modeling import *
from modeling.track import *
from modeling.trajectory import *
from controllers.io_linearization import DFBL, FBL
from simulation.plotting import animate

robot = DifferentialDrive()
track = Track(freq=0.03)
circle = Circle(freq = 0.05)
controller = FBL(kp=[1,1],kd=[1,1], b = 0.05)

loop = Simulation(dt=0.08, robot=robot, controller=controller, reference=track)

q_traj, u_traj, ref_traj = loop.run(
    T = 10*ca.pi, 
    q0 = np.array([0,0,np.pi/6]),
    qd0=np.array([0,0,np.pi/6]))

animation = animate(q_traj, u_traj, ref_traj, robot, track)
import sys
sys.path.append("..")

from simulation.simulation import *
from modeling import *
from modeling.trajectory import *
from controllers.io_linearization import *

robot = DifferentialDrive(); q0 = np.array([0,0,np.pi/6])
# robot = Bicycle(); q0 = np.array([0,0,0,np.pi/6])

controller = FBL(b = 0.1)
controller.set_gains(kp=[1,1],kd=[1,1])
loop = Simulation(robot, controller, dt=0.1)
reference = Circle(freq = 0.05)

# The output will be a state/input trajectory
q_traj, u_traj, ref_traj = loop.run(reference=reference, T = 10*ca.pi, q0 = q0)
from simulation.plotting import animate
animation = animate(q_traj, u_traj, ref_traj, robot)
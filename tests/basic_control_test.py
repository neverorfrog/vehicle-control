import sys
sys.path.append("..")

from simulation.simulation import *
from modeling import *
from modeling.trajectory import *
from controllers.controller import *

# robot = DifferentialDrive()
# q0 = np.array([0,0,np.pi/6])
robot = Bicycle()
q0 = np.array([0,0,0,np.pi/6])

controller = Controller()
loop = Simulation(robot, controller, dt=0.01)
reference = Circle()

# The output will be a state/input trajectory
q_traj, u_traj = loop.run(reference=reference, T = 2*ca.pi, q0 = q0)
from simulation.plotting import animate
animation = animate(q_traj, u_traj, robot)
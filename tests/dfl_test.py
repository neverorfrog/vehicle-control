import sys
sys.path.append("..")

from simulation.simulation import *
from modeling import *
from modeling.trajectory import *
from controllers.dfl import DFL

robot = DifferentialDrive()
controller = DFL()
controller.set_gains(kp=[1,1],kd=[1,1])
loop = Simulation(robot, controller, dt=0.05)
reference = Circle(freq=0.1)

# The output will be a state/input trajectory
q_traj, u_traj = loop.run(reference=reference, T = 10*ca.pi)
from simulation.plotting import animate
animation = animate(q_traj, u_traj, state_labels=['x','y','theta'], input_labels=['v','w'])
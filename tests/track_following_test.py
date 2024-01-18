import sys
sys.path.append("..")

from simulation.simulation import *
from modeling import *
from modeling.track import *
from controllers.dfl import DFL

robot = Bicycle()
q0 = np.array([0,0,0,np.pi/6])

robot = DifferentialDrive()
q0 = np.array([0,0,np.pi/6])

controller = Controller()
controller.set_gains(kp=[1,1],kd=[1,1])
loop = Simulation(robot, controller, dt=0.05)
reference = Track()

# TODO complete test (track reference)

# The output will be a state/input trajectory
q_traj, u_traj = loop.run(reference=reference, T = 10*ca.pi, q0 = q0)
from simulation.plotting import animate
animation = animate(q_traj, u_traj, robot, reference)
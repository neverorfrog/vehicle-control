import sys
sys.path.append("..")

from controllers.control import *
from simulation import *
from controllers.trajectory import *
from controllers.dfl import DFL

model = DifferentialDrive()
loop = DFL(model, dt=0.1)
reference = Circle(freq=0.1)

# The output will be a state/input trajectory
loop.set_gains(kp=[1,1],kd=[1,1])
q_traj, u_traj = loop.run(reference=reference, T = 10*ca.pi)
from simulation.utils import animate
animation = animate(q_traj, u_traj, state_labels=['x','y','theta'], input_labels=['v','w'])
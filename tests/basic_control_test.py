import sys
sys.path.append("..")

from controllers.control import *
from simulation import *
from controllers.trajectory import *

model = DifferentialDrive()
loop = Control(model, dt=0.1)
reference = Circle()

# The output will be a state/input trajectory
q_traj, u_traj = loop.run(reference=reference, T = 2*ca.pi)
from simulation.utils import animate
animation = animate(q_traj, u_traj, state_labels=['x','y','theta'], input_labels=['v','w'])
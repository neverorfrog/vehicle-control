import sys
sys.path.append("..")

from simulation.simulation import *
from modeling import *
from modeling.trajectory import *
from controllers.controller import *

model = DifferentialDrive()
# model = Bicycle()
controller = Controller()
loop = Simulation(model, controller, dt=0.1)
reference = Circle()

# The output will be a state/input trajectory
q_traj, u_traj = loop.run(reference=reference, T = 2*ca.pi)
from simulation.plotting import animate
animation = animate(q_traj, u_traj, model)
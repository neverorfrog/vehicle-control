import sys
sys.path.append(".")

from simulation.trajectory_tracking import TrajectoryTrackingSimulation
from model.differential_drive import DifferentialDrive
from utils.common_utils import *
from environment.trajectory import Circle
from controller.io_linearization import FBL, DFBL

reference = Circle()

# Bicycle model and corresponding controller
robot_config = load_config(f"config/model/differential_drive.yaml")
robot = DifferentialDrive(config=robot_config)
robot.input.v = 0.1
# controller = FBL(kp=np.array([1,1]),kd=np.array([1,1]))
controller = DFBL(kp=np.array([5,5]),kd=np.array([2,2]))

# Simulation
simulation = TrajectoryTrackingSimulation("boh",robot,controller,reference)   
simulation.run(N = 200, animate = True)
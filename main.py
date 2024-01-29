import sys
sys.path.append(".")

from model.state import KinematicCarState
from environment.track import Track
from model.kinematic_car import KinematicCar
from simulation.simulator import RacingSimulation
from controller.kinematic_mpc import RacingMPC
from utils.utils import *
from matplotlib import pyplot as plt
from scipy.integrate import quad

# Track Loading
track_name = "ippodromo"
# track_name = "complicato"
config = load_config(f"config/kinematic_{track_name}.yaml")
track = Track(wp_x=config['wp_x'], 
              wp_y=config['wp_y'], 
              resolution=config['resolution'],
              smoothing=config['smoothing'],
              width=config['width'])

# def test_curvature(s):
#     print(f"x: {track.x(s)}, y: {track.y(s)}")
#     print(f"curv: {track.get_curvature(s)}")
# test_curvature(3.1)
# track.plot(plt.gca())
# plt.show()

# Bicycle model
car = KinematicCar(track, length=0.2, dt=config['model_dt'])
car.state = KinematicCarState(v = 0.5)

# MPC controller
controller = RacingMPC(car=car, config=config)

# Simulation
simulation = RacingSimulation(track_name,car,controller)   
simulation.run()
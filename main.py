import sys
sys.path.append(".")

from model.dynamic_car import DynamicCar, DynamicCarState
from environment.track import Track
from model.kinematic_car import KinematicCar, KinematicState
from simulation.racing import RacingSimulation
from controller.kinematic_mpc import KinematicMPC
from controller.singletrack_mpc import SingleTrackMPC
from utils.common_utils import *
from enum import Enum

class CarType(Enum):
    DYN = "dynamic_car"
    KIN = "kinematic_car"
    
class TrackType(Enum):
    I = "ippodromo"
    C = "complicato"
    
mode = CarType.DYN
track_name = TrackType.I.value

# Track Loading
track_config = load_config(f"config/environment/{track_name}.yaml")
track = Track(wp_x=track_config['wp_x'], 
              wp_y=track_config['wp_y'], 
              resolution=track_config['resolution'],
              smoothing=track_config['smoothing'],
              width=track_config['width'])

# Bicycle model and corresponding controller
car_config = load_config(f"config/model/{mode.value}.yaml")
controller_config = load_config(f"config/controller/{mode.value}_{track_name}.yaml")
if mode is CarType.KIN:
    car = KinematicCar(config=car_config, track = track)
    car.state = KinematicState(v = 1, s = 170)
    controller = KinematicMPC(car=car, config=controller_config)
elif mode is CarType.DYN:
    car = DynamicCar(config=car_config, track = track)
    car.state = DynamicCarState(Ux = 3, s = 50)
    controller = SingleTrackMPC(car=car, config=controller_config)

# Simulation
simulation = RacingSimulation(f"{mode.value}_{track_name}",car,controller)   
simulation.run(N = 1000)
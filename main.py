import sys
sys.path.append(".")

from model.dynamic_car import DynamicCar
from model.state import DynamicCarState, KinematicCarState
from environment.track import Track
from model.kinematic_car import KinematicCar
from simulation.simulator import RacingSimulation
from controller.kinematic_mpc import KinematicMPC
from controller.dynamic_mpc import DynamicMPC
from utils.utils import *
from enum import Enum

class Mode(Enum):
    DYN = "dynamic"
    KIN = "kinematic"
    
mode = Mode.KIN
track_name = "ippodromo"

# Track Loading
track_config = load_config(f"config/environment/{track_name}.yaml")
track = Track(wp_x=track_config['wp_x'], 
              wp_y=track_config['wp_y'], 
              resolution=track_config['resolution'],
              smoothing=track_config['smoothing'],
              width=track_config['width'])

# Bicycle model and corresponding controller
model_config = load_config(f"config/model/{mode.value}.yaml")
controller_config = load_config(f"config/controller/{mode.value}_{track_name}.yaml")
if mode is Mode.KIN:
    car = KinematicCar(track, config=model_config)
    car.state = KinematicCarState(v = 0.5)
    controller = KinematicMPC(car=car, config=controller_config)
elif mode is Mode.DYN:
    car = DynamicCar(track, config=model_config)
    car.state = DynamicCarState(Ux = 0.5, delta = -0.6)
    controller = DynamicMPC(car=car, config=controller_config)

# Simulation
simulation = RacingSimulation(track_name,car,controller)   
simulation.run(N = 194, animate = True)
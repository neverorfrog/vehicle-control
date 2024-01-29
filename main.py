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
from scipy.integrate import quad
from enum import Enum

class Mode(Enum):
    DYN = "dynamic"
    KIN = "kinematic"
    
mode = Mode.KIN

# Track Loading
track_name = f"{mode.value}_ippodromo"
config = load_config(f"config/{track_name}.yaml")
track = Track(wp_x=config['wp_x'], 
              wp_y=config['wp_y'], 
              resolution=config['resolution'],
              smoothing=config['smoothing'],
              width=config['width'])

# Bicycle model and corresponding controller
if mode is Mode.KIN:
    car = KinematicCar(track, length=0.2, dt=config['model_dt'])
    car.state = KinematicCarState(v = 0.5)
    controller = KinematicMPC(car=car, config=config)
elif mode is Mode.DYN:
    car = DynamicCar(track, length=0.2, dt=config['model_dt'])
    car.state = DynamicCarState(Ux = 0.5)
    controller = DynamicMPC(car=car, config=config)

# Simulation
simulation = RacingSimulation(track_name,car,controller)   
simulation.run(N = 88)
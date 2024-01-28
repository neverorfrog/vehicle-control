import sys
sys.path.append(".")

from model.state import KinematicCarState
from environment.track import Track
from model.kinematic_car import KinematicCar
from simulation.simulator import RacingSimulation
from controller.kinematic_mpc import RacingMPC
from utils.utils import *

# Track Loading
track_name = "ippodromo"
track_name = "complicato"
config = load_config(f"config/kinematic_{track_name}.yaml")
track = Track(wp_x=config['wp_x'], 
              wp_y=config['wp_y'], 
              resolution=config['resolution'],
              smoothing=config['smoothing'],
              width=config['width'])

# Bicycle model
car = KinematicCar(track, length=0.2, dt=config['model_dt'])
s0 = config['initial_state']
car.state = KinematicCarState(x = s0['x'], y = s0['y'], v = s0['v'])

# MPC controller
controller = RacingMPC(car=car, config=config)

# Simulation
simulation = RacingSimulation(track_name,car,controller)   
simulation.run()#N = config['n_steps'])
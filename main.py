import sys
sys.path.append(".")

from environment.track import Track
from models.kinematic_car import KinematicCar, KinematicState
from models.dynamic_car import DynamicCar, DynamicCarState
from models.dynamic_point_mass import DynamicPointMass, DynamicPointMassState
from simulation.racing import RacingSimulation
from controllers.mpc.kinematic_mpc import KinematicMPC
from controllers.mpc.dynamic_mpc import DynamicMPC
from utils.common_utils import load_config
from enum import Enum
from controllers.mpc.pointmass_mpc import PointMassMPC

class CarType(Enum):
    KIN = "kinematic_car"
    DYN = "dynamic_car"
    DPM = "dynamic_point_mass"
    CAS = "cascaded"
    
class TrackType(Enum):
    I = "ippodromo"
    C = "complicato"
    
car_type = CarType.KIN
track_name = TrackType.I.value

# Track Loading
track_config = load_config(f"config/environment/{track_name}.yaml")
track = Track(wp_x=track_config['wp_x'], 
              wp_y=track_config['wp_y'], 
              resolution=track_config['resolution'],
              smoothing=track_config['smoothing'],
              width=track_config['width'])

# Bicycle model and corresponding controller
car_config = load_config(f"config/models/{car_type.value}.yaml")
controller_config = load_config(f"config/controllers/{track_name}/{car_type.value}.yaml")
if car_type is CarType.KIN:
    car = KinematicCar(config=car_config, track = track)
    car.state = KinematicState(v = 1, s = 170)
    controller = KinematicMPC(car=car, config=controller_config)
elif car_type is CarType.DYN:
    car = DynamicCar(config=car_config, track = track)
    car.state = DynamicCarState(Ux = 3, s = 50)
    controller = DynamicMPC(car=car, config=controller_config)
elif car_type is CarType.DPM:
    car = DynamicPointMass(config=car_config, track = track)
    car.state = DynamicPointMassState(V = 3, s = 50)
    controller = PointMassMPC(car=car, config=controller_config)

# Simulation
simulation = RacingSimulation(f"{car_type.value}_{track_name}",car,controller)   
simulation.run(N = 1000)
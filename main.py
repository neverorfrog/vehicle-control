import sys
sys.path.append(".")

from environment.track import Track
from models.kinematic_car import KinematicCar, KinematicState
from models.dynamic_car import DynamicCar, DynamicCarState
from models.dynamic_point_mass import DynamicPointMass, DynamicPointMassState
from simulation.racing import RacingSimulation
from controllers.mpc.kinematic_mpc import KinematicMPC
from controllers.mpc.dynamic_mpc import DynamicMPC
from controllers.mpc.cascaded_mpc import CascadedMPC
from utils.common_utils import load_config, ControlType, CarType, TrackType
from controllers.mpc.pointmass_mpc import PointMassMPC

# Configuration
control_type = ControlType.CAS 
car_type = CarType.DYN
track_name = TrackType.N.value

# Track Definition
track_config = load_config(f"config/environment/{track_name}.yaml")
track = Track(wp_x=track_config['wp_x'], 
              wp_y=track_config['wp_y'], 
              resolution=track_config['resolution'],
              smoothing=track_config['smoothing'],
              width=track_config['width'])

# Model and Controller Definition
if control_type is ControlType.CAS:
    controller_config = load_config(f"config/controllers/{track_name}/cascaded_{track_name}.yaml")
    #DYNAMIC CAR
    car_config = load_config(f"config/models/dynamic_car.yaml")
    car = DynamicCar(config=car_config, track=track)
    car.state = DynamicCarState(Ux = 4, s = 1)
    #DYNAMIC POINT MASS
    pm_config = load_config(f"config/models/dynamic_point_mass.yaml")
    point_mass = DynamicPointMass(config=pm_config, track=track)
    point_mass.state = DynamicPointMassState()
    controller = CascadedMPC(car=car, point_mass=point_mass, config=controller_config)
    simulation = RacingSimulation(f"cascaded_{track_name}",car,point_mass,controller)     
elif control_type is ControlType.SIN:
    controller_config = load_config(f"config/controllers/{track_name}/{car_type.value}_{track_name}.yaml")
    car_config = load_config(f"config/models/{car_type.value}.yaml")
    point_mass = None
    if car_type is CarType.KIN:
        car = KinematicCar(config=car_config, track = track)
        car.state = KinematicState(v = 1, s = 30)
        controller = KinematicMPC(car=car, config=controller_config)
    elif car_type is CarType.DYN:
        car = DynamicCar(config=car_config, track = track)
        car.state = DynamicCarState(Ux = 4, s = 30)
        controller = DynamicMPC(car=car, config=controller_config)
    elif car_type is CarType.DPM:
        car = DynamicPointMass(config=car_config, track = track)
        car.state = DynamicPointMassState(V = 3, s = 30)
        controller = PointMassMPC(car=car, config=controller_config)
    simulation = RacingSimulation(f"{car_type.value}_{track_name}",car,point_mass,controller)  

simulation.run(N = 500)
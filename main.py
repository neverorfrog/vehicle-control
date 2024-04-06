import sys
sys.path.append(".")

import controllers
import models
import environment as env
from simulation.racing import RacingSimulation
from utils.common_utils import load_config, ControlType, CarType, TrackType
from matplotlib import pyplot as plt
import logging

# Configuration
control_type = ControlType.CAS 
car_type = CarType.DYN
track_name = TrackType.C.value

# Track Definition
track_config = load_config(f"config/environment/{track_name}.yaml")
track = env.Track(wp_x=track_config['wp_x'], 
              wp_y=track_config['wp_y'], 
              resolution=track_config['resolution'],
              smoothing=track_config['smoothing'],
              width=track_config['width'])

# for waypoint in track.waypoints:
#     print(waypoint)
# track.plot(plt.gca())
# plt.show()
# exit()

# Model and Controller Definition
if control_type is ControlType.CAS:
    controller_config = load_config(f"config/controllers/{track_name}/cascaded_{track_name}.yaml")
    #DYNAMIC CAR
    car_config = load_config(f"config/models/dynamic_car.yaml")
    car = models.DynamicCar(config=car_config, track=track)
    car.state = models.DynamicCarState(Ux = 4, s = 1)
    #DYNAMIC POINT MASS
    pm_config = load_config(f"config/models/dynamic_point_mass.yaml")
    point_mass = models.DynamicPointMass(config=pm_config, track=track)
    point_mass.state = models.DynamicPointMassState()
    controller = controllers.CascadedMPC(car=car, point_mass=point_mass, config=controller_config)
    simulation = RacingSimulation(f"cascaded_{track_name}",car,point_mass,controller)     
elif control_type is ControlType.SIN:
    controller_config = load_config(f"config/controllers/{track_name}/{car_type.value}_{track_name}.yaml")
    car_config = load_config(f"config/models/{car_type.value}.yaml")
    point_mass = None
    if car_type is CarType.KIN:
        car = models.KinematicCar(config=car_config, track = track)
        car.state = models.KinematicState(v = 1, s = 30)
        controller = controllers.KinematicMPC(car=car, config=controller_config)
    elif car_type is CarType.DYN:
        car = models.DynamicCar(config=car_config, track = track)
        car.state = models.DynamicCarState(Ux = 4, s = 30)
        controller = controllers.DynamicMPC(car=car, config=controller_config)
    elif car_type is CarType.DPM:
        car = models.DynamicPointMass(config=car_config, track = track)
        car.state = models.DynamicPointMassState(V = 3, s = 30)
        controller = controllers.PointMassMPC(car=car, config=controller_config)
    simulation = RacingSimulation(f"{car_type.value}_{track_name}",car,point_mass,controller)  

logfile = '/home/neverorfrog/code/vehicle-control-cascaded-models/logs/test.log'
with open(logfile, "w") as f:
    # sys.stdout = f
    print(f"Car configuration: {car_config}")
    print(f"Controller configuration: {controller_config}")
    simulation.run(N = 500)
sys.stdout = sys.__stdout__
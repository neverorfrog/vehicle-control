import sys
import os
sys.path.append(".")

import controllers
import models
import environment as env
from simulation.racing import RacingSimulation
from utils.common_utils import load_config, ControlType, CarType, TrackType
import matplotlib.pyplot as plt

# ======== Configuration ============
control_type = ControlType.CAS 
car_type = CarType.DYN
track_name = TrackType.I.value

# =========== Track Definition ============
track_config = load_config(f"config/environment/{track_name}.yaml")
track = env.Track(corners=track_config['corners'],
                  smoothing=track_config['smoothing'],
                  resolution=track_config['resolution'],
                  width=track_config['width'])

# ========= Model Definition ==============

#DYNAMIC CAR
car_config = load_config(f"config/models/dynamic_car.yaml")
car = models.DynamicCar(config=car_config, track=track)
car.state = models.DynamicCarState(Ux = 4, s = 1)
#DYNAMIC POINT MASS
pm_config = load_config(f"config/models/dynamic_point_mass.yaml")
point_mass = models.DynamicPointMass(config=pm_config, track=track)
point_mass.state = models.DynamicPointMassState()

# ============ Controller Definition ================
controller_config = load_config(f"config/controllers/{track_name}/cascaded_{track_name}.yaml")
controller = controllers.CascadedMPC(car=car, point_mass=point_mass, config=controller_config)
if controller.M > 0:
    simulation = RacingSimulation(f"cascaded_{track_name}",car,point_mass,controller)  
else:
    simulation = RacingSimulation(f"singletrack_{track_name}",car,point_mass,controller)   


src_dir = os.path.dirname(os.path.abspath(__file__))
logfile = f'simulation/logs/{simulation.name}.log'
with open(logfile, "w") as f:
    sys.stdout = f
    print(f"Car configuration: {car_config}")
    print(f"Controller configuration: {controller_config}")
    simulation.run(N = 500)
sys.stdout = sys.__stdout__
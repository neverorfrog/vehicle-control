import sys
import os
sys.path.append(".")

import controllers
import models
import environment as env
from simulation.racing import RacingSimulation
from utils.common_utils import load_config, CarType, TrackType
from omegaconf import OmegaConf

# ======== Configuration =========================
car_type = CarType.KIN
track_name = TrackType.I.value

# =========== Track Definition ====================
track_config = OmegaConf.create(load_config(f"config/environment/{track_name}.yaml"))
OmegaConf.set_readonly(track_config, True)
OmegaConf.set_struct(track_config, True)
track = env.Track(
    corners=track_config.corners,
    smoothing=track_config.smoothing,
    resolution=track_config.resolution,
    width=track_config.width,
    obstacle_data=track_config.obstacle_data
)

# ========= Model Definition =======================
if car_type == CarType.DYN:
    car_config = OmegaConf.create(load_config(f"config/models/dynamic_car.yaml"))
    car = models.DynamicCar(config=car_config, track=track)
    car.state = models.DynamicCarState(Ux = 4, s = 1)
    point_mass = models.DynamicPointMass(config=car_config, track=track)
    point_mass.state = models.DynamicPointMassState()
elif car_type == CarType.KIN:
    car_config = OmegaConf.create(load_config(f"config/models/kinematic_car.yaml"))
    car = models.KinematicCar(config=car_config, track=track)
    car.state = models.KinematicCarState(v=1)
OmegaConf.set_readonly(car_config, True)
OmegaConf.set_struct(car_config, True)


# ============ Controller Definition ================

if car_type == CarType.KIN:
    name = f"kinematic_{track_name}"
    point_mass = None
else:
    name = f"cascaded_{track_name}"
controller_config = OmegaConf.create(load_config(f"config/controllers/{track_name}/{name}.yaml"))
OmegaConf.set_readonly(controller_config, True)
OmegaConf.set_struct(controller_config, True)

if car_type == CarType.KIN:
    controller = controllers.KinematicMPC(car=car, config=controller_config)
else:
    controller = controllers.CascadedMPC(car=car, point_mass=point_mass, config=controller_config)
    if controller.M == 0:
        name = f"singletrack_{track_name}"
if controller.config.obstacles:
    name += "_obstacles"

# ============ Simulation ============================
simulation = RacingSimulation(name,car,controller,point_mass)
src_dir = os.path.dirname(os.path.abspath(__file__))
logfile = f'simulation/logs/{simulation.name}.log'
with open(logfile, "w") as f:
    # sys.stdout = f
    print(f"Car configuration: {car_config}")
    print(f"Controller configuration: {controller_config}")
    simulation.run(N = 500)
sys.stdout = sys.__stdout__
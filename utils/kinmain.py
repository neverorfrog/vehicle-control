import sys
import os
sys.path.append(".")

import controllers
import models
import environment as env
import simulation as sim
from utils.common_utils import load_config, ControlType, CarType, TrackType
from omegaconf import OmegaConf
import casadi as ca

# ======== Configuration =========================
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
car_config = OmegaConf.create(load_config(f"config/models/kinematic_car.yaml"))
car = models.KinematicCar(config=car_config, track=track)
car.state = models.KinematicCarState(v=0.1)

# ============ Controller Definition ================
controller_config = OmegaConf.create(load_config(f"config/controllers/{track_name}/kinematic_{track_name}.yaml"))
OmegaConf.set_readonly(controller_config, True)
OmegaConf.set_struct(controller_config, True)
controller = controllers.KinematicMPC(car=car, config=controller_config)
name = f"kinematic_{track_name}"
simulation = sim.RacingSimulation(name=name,car=car,controller=controller)

# ============ Simulation ============================
src_dir = os.path.dirname(os.path.abspath(__file__))
logfile = f'simulation/logs/{simulation.name}.log'
with open(logfile, "w") as f:
    # sys.stdout = f
    print(f"Car configuration: {car_config}")
    print(f"Controller configuration: {controller_config}")
    simulation.run(N = 500)
sys.stdout = sys.__stdout__
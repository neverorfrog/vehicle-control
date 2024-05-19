import controllers as control
import models
import environment as env
import utils.common_utils as utils
from simulation import RacingSimulation
from omegaconf import OmegaConf

# ======== Configuration ========================================================
track_name = utils.TrackType.I.value
names = []
names.append("singletrack")
# names.append("cascaded")
sim_name = f"race1_obstacles_{track_name}"
# sim_name = f"cascaded_obstacles2_{track_name}"
# sim_name = f"singletrack_{track_name}"

# =========== Track Definition ===================================================
track_config = OmegaConf.create(utils.load_config(f"config/environment/{track_name}.yaml"))
OmegaConf.set_readonly(track_config, True)
OmegaConf.set_struct(track_config, True)
track = env.Track(
    name=track_name,
    corners=track_config.corners,
    smoothing=track_config.smoothing,
    resolution=track_config.resolution,
    width=track_config.width,
    obstacle_data=track_config.obstacle_data
)

# ========= Models Definition ======================================================
car_config = OmegaConf.create(utils.load_config(f"config/models/dynamic_car.yaml"))
cars = [models.DynamicCar(config=car_config, track=track) for _ in names]
point_masses = [models.DynamicPointMass(config=car_config, track=track) for _ in names]
for car in cars: car.state = models.DynamicCarState(Ux = 4, s = 1)

# ============ Controller Definition ================================================
controller_configs = [OmegaConf.create(utils.load_config(f"config/controllers/{track_name}/{name}_{track_name}.yaml")) for name in names]
combriccola = zip(cars, point_masses, controller_configs)
controllers = [control.CascadedMPC(car=car, point_mass=point_mass, config=config) for car,point_mass,config in combriccola]

# ============ Run Simulation ======================================================
colors = [controller.config.color for controller in controllers]
simulation = RacingSimulation(sim_name,names,cars,controllers,colors,track,load=False)
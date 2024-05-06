import sys
import os
from matplotlib import pyplot as plt
from matplotlib.backend_bases import FigureManagerBase
sys.path.append(".")
import controllers as control
import models
import environment as env
import utils.common_utils as utils
from simulation.racing import RacingSimulation
from omegaconf import OmegaConf

# ======== Configuration ========================================================
track_name = utils.TrackType.I.value
names = []
names.append("cascaded")
# names.append("singletrack")
# sim_name = f"race_obstacles_{track_name}"
sim_name = f"cascaded_{track_name}"
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
# cascaded models
car_config = OmegaConf.create(utils.load_config(f"config/models/dynamic_car.yaml"))
cars = [models.DynamicCar(config=car_config, track=track) for _ in names]
point_masses = [models.DynamicPointMass(config=car_config, track=track) for _ in names]
for car in cars: car.state = models.DynamicCarState(Ux = 4, s = 1.5)

#kinematic bicycle models
kincar_config = OmegaConf.create(utils.load_config(f"config/models/kinematic_car.yaml"))
kin_cars = [models.KinematicCar(config=kincar_config, track=track) for _ in names]
for kincar in kin_cars: kincar.state = models.KinematicCarState(v=0.1)

# ============ Controller Definition ================================================
controller_configs = [OmegaConf.create(utils.load_config(f"config/controllers/{track_name}/{name}_{track_name}.yaml")) for name in names]
kincontroller_config = OmegaConf.create(utils.load_config(f"config/controllers/{track_name}/kinematic_{track_name}.yaml"))
kin_controllers = [control.KinematicMPC(car=car, config=kincontroller_config) for car in kin_cars]
combriccola = zip(cars, point_masses, kin_controllers, controller_configs)
controllers = [control.CascadedMPC(car=car, point_mass=point_mass, kin_controller=kin_controller, config=config) for car,point_mass,kin_controller,config in combriccola]

# ============ Run Simulation ======================================================
simulation = RacingSimulation(names,cars,controllers,track)
src_dir = os.path.dirname(os.path.abspath(__file__))
logfile = f'simulation/logs/{sim_name}.log'
with open(logfile, "w") as f:
    sys.stdout = f
    state_traj, action_traj, preds, elapsed = simulation.run(N=500)

# ============ Show Animation ======================================================
animation = simulation.animate(state_traj, action_traj, preds, elapsed) 
fig_manager: FigureManagerBase = plt.get_current_fig_manager()
fig_manager.window.showMaximized()
plt.show(block=True)

# =========== Save Data and Animation ==============================================
# simulation.save(state_traj, action_traj, preds, elapsed)
# animation.save(f"simulation/videos/{sim_name}.gif",fps=13, dpi=200, writer='pillow')
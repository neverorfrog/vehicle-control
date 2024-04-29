import sys
import os
from matplotlib import pyplot as plt
from matplotlib.backend_bases import FigureManagerBase
sys.path.append(".")
os.environ["XDG_SESSION_TYPE"] = "xcb"
import controllers as control
import models
import environment as env
import utils.common_utils as utils
from simulation.racing import RacingSimulation
from omegaconf import OmegaConf

# ======== Configuration =========================
track_name = utils.TrackType.I.value
names = []
names.append("cascaded")
names.append("singletrack")
sim_name = f"race_{track_name}"
# sim_name = f"cascaded_{track_name}"
# sim_name = f"singletrack_{track_name}"

# =========== Track Definition ====================
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

# ========= Models Definition =======================
car_config = OmegaConf.create(utils.load_config(f"config/models/dynamic_car.yaml"))
cars = [models.DynamicCar(config=car_config, track=track) for _ in names]
for car in cars: car.state = models.DynamicCarState(Ux = 4, s = 1)
point_mass = models.DynamicPointMass(config=car_config, track=track)
point_mass.state = models.DynamicPointMassState()

# ============ Controller Definition ================
controller_configs = [OmegaConf.create(utils.load_config(f"config/controllers/{track_name}/{name}_{track_name}.yaml")) for name in names]
controllers = [control.CascadedMPC(car=car, point_mass=point_mass, config=config) for config in controller_configs]

# ============ Simulation ============================
simulation = RacingSimulation(names,cars,controllers,track)
src_dir = os.path.dirname(os.path.abspath(__file__))
logfile = f'simulation/logs/{sim_name}.log'
with open(logfile, "w") as f:
    sys.stdout = f
    state_traj, action_traj, preds, elapsed = simulation.run(N=500)
    simulation.save(state_traj, action_traj, preds, elapsed)
animation = simulation.animate(state_traj, action_traj, preds, elapsed) 
fig_manager: FigureManagerBase = plt.get_current_fig_manager()
fig_manager.window.showMaximized()
plt.show(block=True)
animation.save(f"simulation/videos/{sim_name}.gif",fps=20, dpi=200, writer='pillow')
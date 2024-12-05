from omegaconf import OmegaConf

import vehicle_control.utils.common_utils as utils
from vehicle_control.simulation import KinematicRacingSimulator

simconfig = OmegaConf.create(utils.load_config("config/simconfig.yaml"))
trackconfig = OmegaConf.create(
    utils.load_config(f"config/environment/{simconfig.track_name}.yaml")
)
carconfig = OmegaConf.create(utils.load_config("config/models/kinematic_car.yaml"))
simulator = KinematicRacingSimulator(simconfig, carconfig, trackconfig)
# simulator.summarize()
simulator.run()

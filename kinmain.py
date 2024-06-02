import utils.common_utils as utils
from simulation import KinematicRacingSimulator
from omegaconf import OmegaConf

simconfig = OmegaConf.create(utils.load_config(f"config/simconfig.yaml"))
trackconfig = OmegaConf.create(utils.load_config(f"config/environment/{simconfig.track_name}.yaml"))
carconfig = OmegaConf.create(utils.load_config(f"config/models/kinematic_car.yaml"))
simulator = KinematicRacingSimulator(simconfig, carconfig, trackconfig)
# simulator.summarize()
simulator.run()
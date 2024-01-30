import sys
sys.path.append("..")

from model.dynamic_car import DynamicCar
from utils.utils import load_config
from environment.track import Track

config = load_config("../config/kinematic_ippodromo.yaml")

track = Track(wp_x=config['wp_x'], 
              wp_y=config['wp_y'], 
              resolution=config['resolution'],
              smoothing=config['smoothing'],
              width=config['width'])

car = DynamicCar(track=track, length=0.4, dt=config['model_dt'])
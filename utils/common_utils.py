from enum import Enum
import numpy as np
import yaml

class TrackType(Enum):
    S = "standard"
    I = "ippodromo"
    B = "bean"
    G = "gokart"

# Load configuration from a YAML file
def load_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def wrap(angle):
    '''Wrap between -pi and pi'''
    if angle < -np.pi:
        w_angle = 2 * np.pi + angle
    elif angle > np.pi:
        w_angle = angle - 2 * np.pi
    else:
        w_angle = angle

    return w_angle
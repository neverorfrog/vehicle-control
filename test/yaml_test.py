import sys
import os
sys.path.append("..")

from utils.utils import load_config

config_path = "../config/ippodromo.yaml"

config = load_config(config_path)

print(config['state_weights'])


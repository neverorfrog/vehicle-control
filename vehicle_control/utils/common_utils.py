import os
from enum import Enum

import numpy as np
import yaml


class TrackType(Enum):
    S = "shoe"
    I = "ippodromo"  # noqa: E741
    B = "bean"
    G = "gokart"


# Load configuration from a YAML file
def load_config(file_path):
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def wrap(angle):
    """Wrap between -pi and pi"""
    if angle < -np.pi:
        w_angle = 2 * np.pi + angle
    elif angle > np.pi:
        w_angle = angle - 2 * np.pi
    else:
        w_angle = angle

    return w_angle


def project_root() -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    max_iterations = 100  # Set a limit for the number of iterations
    for _ in range(max_iterations):
        if (
            "requirements.txt" in os.listdir(current_dir)
            or "setup.py" in os.listdir(current_dir)
            or "pyproject.toml" in os.listdir(current_dir)
        ):
            return current_dir
        current_dir = os.path.dirname(current_dir)
    raise FileNotFoundError(
        "requirements.txt not found in any parent directories within the iteration limit"
    )

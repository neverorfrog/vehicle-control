import numpy as np
import yaml

# Load configuration from a YAML file
def load_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def integrate(q,u,curvature,ode,h):
    '''
    RK4 integrator
    h: integration interval
    '''
    qd_1 = ode(q, u, curvature)
    qd_2 = ode(q + (h/2)*qd_1, u, curvature)
    qd_3 = ode(q + (h/2)*qd_2, u, curvature)
    qd_4 = ode(q + h*qd_3, u, curvature)
    newq = q + (1/6) * (qd_1 + 2 * qd_2 + 2 * qd_3 + qd_4) * h
    return newq

def wrap(angle):
    '''Wrap between -pi and pi'''
    if angle < -np.pi:
        w_angle = 2 * np.pi + angle
    elif angle > np.pi:
        w_angle = angle - 2 * np.pi
    else:
        w_angle = angle

    return w_angle

def sign(a):
    return 1 if a >= 0 else -1
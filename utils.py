import inspect
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import casadi as ca

class Parameters:
    def save_parameters(self, ignore=[]):
        """Save function arguments into class attributes"""
        frame = inspect.currentframe().f_back
        _, _, _, local_vars = inspect.getargvalues(frame)
        self.hparams = {k:v for k, v in local_vars.items()
                        if k not in set(ignore+['self']) and not k.startswith('_')}
        for k, v in self.hparams.items():
            setattr(self, k, v)
            
def RK4(state, input, state_dot, dt, integration_steps=1):
    '''
    RK4 integrator
    dt:             integration interval
    N_steps:        number of integration steps per integration interval, default:1
    '''
    h = dt/integration_steps
    current_state = state
    transition_function = ca.Function('xdot', [state, input], [state_dot])

    for _ in range(integration_steps):
        k_1 = transition_function(current_state, input)
        k_2 = transition_function(current_state + (dt/2)*k_1,input)
        k_3 = transition_function(current_state + (dt/2)*k_2,input)
        k_4 = transition_function(current_state + dt*k_3,input)

        current_state += (1/6) * (k_1 + 2 * k_2 + 2 * k_3 + k_4) * h

    return current_state

def plot_trajectory(state_traj, input_traj, state_labels, input_labels):
    
    plt.figure(figsize=(3, 10))

    N = input_traj.shape[1]
    ts = np.arange(0, N+1)
    state_len = state_traj.shape[0]
    input_len = input_traj.shape[0]

    for j in range(state_len):
        plt.subplot(state_len+input_len, 1, j+1)
        plt.plot(ts, state_traj[j, :].T, '-', alpha=0.7)
        plt.ylabel(rf'${state_labels[j]}_k$')
        plt.grid()

    for j in range(state_len, state_len+input_len):
        plt.subplot(state_len+input_len, 1, j+1)
        plt.step(ts[:-1], input_traj[j-state_len,:].T, alpha=0.7)
        plt.ylabel(rf'${input_labels[j-state_len]}_k$')
        plt.grid()
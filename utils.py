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
        
import numpy as np
from matplotlib.gridspec import GridSpec

def animate(state_traj, input_traj, state_labels, input_labels):
    # simulation params
    N = input_traj.shape[1]
    state_max = max(state_traj.min(), state_traj.max(), key=abs)
    input_max = max(input_traj.min(), input_traj.max(), key=abs)
    
    # figure params
    grid = GridSpec(2, 2, width_ratios=[3, 1])
    ax_large = plt.subplot(grid[:, 0])
    ax_small1 = plt.subplot(grid[0, 1])
    ax_small2 = plt.subplot(grid[1, 1])
    
    def update(i):
        ax_large.cla()
        ax_large.axis((-5, 5, -2, 2))
        ax_large.set_aspect('equal')
        
        x,y,theta = state_traj[:,i]
        r = 0.1
        
        # Plot circular shape
        circle = plt.Circle(xy=(x,y), radius=r, edgecolor='b', facecolor='none', lw=2)
        ax_large.add_patch(circle)

        # Plot directional tick
        line_length = 1.5 * r
        line_end_x = x + line_length * np.cos(theta)
        line_end_y = y + line_length * np.sin(theta)
        ax_large.plot([x, line_end_x], [y, line_end_y], color='r', lw=2)
        
        ax_small1.cla()
        ax_small1.axis((0, N, -state_max*1.1, state_max*1.1))
        ax_small1.plot(state_traj[:, :i].T, '-', alpha=0.7,label=state_labels)
        ax_small1.legend()

        ax_small2.cla()
        ax_small2.axis((0, N, -input_max*1.1, input_max*1.1))
        ax_small2.plot(input_traj[:, :i].T, '-', alpha=0.7,label=input_labels)
        ax_small2.legend()

    return update
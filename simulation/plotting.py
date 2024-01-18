#inspired by https://github.com/DIAG-Robotics-Lab/underactuated

from collections import deque
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from modeling.kin_model import Model

def animate(state_traj, input_traj, model: Model):
    # simulation params
    N = input_traj.shape[0]
    state_max = max(state_traj.min(), state_traj.max(), key=abs)
    input_max = max(input_traj.min(), input_traj.max(), key=abs)
    
    # figure params
    grid = GridSpec(2, 2, width_ratios=[3, 1])
    ax_large = plt.subplot(grid[:, 0])
    ax_small1 = plt.subplot(grid[0, 1])
    ax_small2 = plt.subplot(grid[1, 1])
    
    # last w trajectory points
    window_size = 100
    window = deque(maxlen=window_size)
    
    def update(i):
        ax_large.cla()
        ax_large.axis((-5, 5, -5, 5))
        ax_large.set_aspect('equal')
        model.plot(ax_large, state_traj[i,:], window)
        ax_large.grid()
        
        ax_small1.cla()
        ax_small1.axis((0, N, -state_max*1.1, state_max*1.1))
        ax_small1.plot(state_traj[:i, :], '-', alpha=0.7,label=model.state_labels)
        ax_small1.legend()

        ax_small2.cla()
        ax_small2.axis((0, N, -input_max*1.1, input_max*1.1))
        ax_small2.plot(input_traj[:i, :], '-', alpha=0.7,label=model.input_labels)
        ax_small2.legend()

    animation = FuncAnimation(fig=plt.gcf(), func=update, frames=N+1, repeat=True, interval=0.01)
    plt.show()  
    return animation
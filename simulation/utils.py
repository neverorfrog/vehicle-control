from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from collections import deque

def animate(state_traj, input_traj, state_labels, input_labels):
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
    x_window = deque(maxlen=window_size)
    y_window = deque(maxlen=window_size)
    
    def update(i):
        ax_large.cla()
        ax_large.axis((-5, 5, -5, 5))
        ax_large.set_aspect('equal')
        ax_large.grid()
        
        x,y,theta = state_traj[i,:]
        
        r = 0.2
        x_window.append(x)
        y_window.append(y)
        
        # Plot circular shape
        circle = plt.Circle(xy=(x,y), radius=r, edgecolor='b', facecolor='none', lw=2)
        ax_large.add_patch(circle)
        # Plot directional tick
        line_length = 1.5 * r
        line_end_x = x + line_length * np.cos(theta)
        line_end_y = y + line_length * np.sin(theta)
        ax_large.plot([x, line_end_x], [y, line_end_y], color='r', lw=3)
        # Plot last window points
        ax_large.plot(x_window,y_window)
        
        ax_small1.cla()
        ax_small1.axis((0, N, -state_max*1.1, state_max*1.1))
        ax_small1.plot(state_traj[:i, :], '-', alpha=0.7,label=state_labels)
        ax_small1.legend()

        ax_small2.cla()
        ax_small2.axis((0, N, -input_max*1.1, input_max*1.1))
        ax_small2.plot(input_traj[:i, :], '-', alpha=0.7,label=input_labels)
        ax_small2.legend()

    animation = FuncAnimation(fig=plt.gcf(), func=update, frames=N+1, repeat=True, interval=0.01)
    plt.show()  
    return animation
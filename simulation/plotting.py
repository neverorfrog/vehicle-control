#inspired by https://github.com/DIAG-Robotics-Lab/underactuated

from collections import deque
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from modeling.robot import Robot
from modeling.track import Track

def animate(state_traj, input_traj, ref_traj, robot: Robot, track: Track = None):
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
    ref_window = deque(maxlen=window_size)
    
    def update(i):
        ax_large.cla()
        ax_large.axis((-5, 5, -5, 5))
        ax_large.set_aspect('equal')
        x,y = robot.plot(ax_large, state_traj[i,:])
        
        # Plot last window points
        window.append([x,y])
        ref_window.append(ref_traj[i,:])
        window_np = np.array(window)
        ref_window_np = np.array(ref_window)
        ax_large.plot(window_np[:,0],window_np[:,1],"k")
        ax_large.plot(ref_window_np[:,0],ref_window_np[:,1],"b")
        ax_large.grid()
        
        # Plot track
        if track is not None:
            track.plot(ax_large)
        
        ax_small1.cla()
        ax_small1.axis((0, N, -state_max*1.1, state_max*1.1))
        ax_small1.plot(state_traj[:i, :], '-', alpha=0.7,label=robot.state_labels)
        ax_small1.legend()

        ax_small2.cla()
        ax_small2.axis((0, N, -input_max*1.1, input_max*1.1))
        ax_small2.plot(input_traj[:i, :], '-', alpha=0.7,label=robot.input_labels)
        ax_small2.legend()

    animation = FuncAnimation(fig=plt.gcf(), func=update, frames=N+1, repeat=True, interval=0.01)
    plt.show()  
    return animation
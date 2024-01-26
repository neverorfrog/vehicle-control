#adapted from https://github.com/DIAG-Robotics-Lab/underactuated
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from modeling.track import Track

def animate(state_traj: list, input_traj: list, state_preds: list = None, robot = None, track: Track = None):
    assert isinstance(state_traj,list), "State trajectory has to be a list"
    assert isinstance(state_traj,list), "Input trajectory has to be a list"

    # simulation params
    N = len(input_traj)
    x_y = np.array(state_traj)[:,:2]
    v_psi_delta = np.array(state_traj)[:,2:5] # taking just v,psi,delta TO
    a_w = np.array(input_traj)
    
    state_max = max(v_psi_delta.min(), v_psi_delta.max(), key=abs)
    input_max = max(a_w.min(), a_w.max(), key=abs)
    # pos_max = max(state_traj[:-1].min(), state_traj[:-1].max(), key=abs)
    
    # figure params
    grid = GridSpec(2, 2, width_ratios=[2, 1])
    ax_large = plt.subplot(grid[:, 0])
    ax_small1 = plt.subplot(grid[0, 1])
    ax_small2 = plt.subplot(grid[1, 1])
    
    def update(i):
        state = state_traj[i]
        
        ax_large.cla()
        ax_large.set_aspect('equal')
        robot.plot(ax_large, state)
        ax_large.plot(x_y[:i, 0],x_y[:i, 1],"k")
        
        # Plot track
        if track is not None:
            track.plot(ax_large, display_drivable_area=False)
            
        # Plot state predictions of MPC
        if state_preds is not None:
            preds = state_preds[i]
            ax_large.plot(preds[0,:], preds[1,:],"go")
            
        
        ax_small1.cla()
        ax_small1.axis((0, N, -state_max*1.1, state_max*1.1))
        ax_small1.plot(v_psi_delta[:i, :], '-', alpha=0.7,label=['v',r'$\psi$',r'$\delta$'])
        ax_small1.legend()

        ax_small2.cla()
        ax_small2.axis((0, N, -input_max*1.1, input_max*1.1))
        ax_small2.plot(a_w[:i, :], '-', alpha=0.7,label=['a','w'])
        ax_small2.legend()

    animation = FuncAnimation(
        fig=plt.gcf(), func=update, 
        frames=N, interval=0.01, 
        repeat=False, repeat_delay=5000
    )
    plt.show()  
    return animation
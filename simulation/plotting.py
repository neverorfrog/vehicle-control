#adapted from https://github.com/DIAG-Robotics-Lab/underactuated
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from modeling.track import Track

def animate(state_traj, input_traj, state_preds = None, robot = None, track: Track = None):
    # simulation params
    N = input_traj.shape[0]
    state_max = max(state_traj.min(), state_traj.max(), key=abs)
    input_max = max(input_traj.min(), input_traj.max(), key=abs)
    # pos_max = max(state_traj[:-1].min(), state_traj[:-1].max(), key=abs)
    
    # figure params
    grid = GridSpec(2, 2, width_ratios=[2, 1])
    ax_large = plt.subplot(grid[:, 0])
    ax_small1 = plt.subplot(grid[0, 1])
    ax_small2 = plt.subplot(grid[1, 1])
    
    def update(i):
        ax_large.cla()
        # ax_large.axis((-pos_max, pos_max, -pos_max, pos_max))
        ax_large.set_aspect('equal')
        robot.plot(ax_large, state_traj[i,:])
        ax_large.plot(state_traj[:i, 0],state_traj[:i, 1],"k")
        # ax_large.grid()
        
        # Plot track
        if track is not None:
            track.plot(ax_large, display_drivable_area=False)
            
        # Plot state predictions of MPC
        if state_preds is not None and i < len(input_traj):
            preds = state_preds[i,:,:]
            ax_large.plot(preds[0,:], preds[1,:],"go")
            
        
        ax_small1.cla()
        ax_small1.axis((0, N, -state_max*1.1, state_max*1.1))
        ax_small1.plot(state_traj[:i, :4], '-', alpha=0.7,label=['x','y',r'$\psi$',r'$\delta$'])
        ax_small1.legend()

        ax_small2.cla()
        ax_small2.axis((0, N, -input_max*1.1, input_max*1.1))
        ax_small2.plot(input_traj[:i, :2], '-', alpha=0.7,label=['v','w'])
        ax_small2.legend()

    animation = FuncAnimation(fig=plt.gcf(), func=update, frames=N+1, repeat=True, interval=0.01)
    plt.show()  
    return animation
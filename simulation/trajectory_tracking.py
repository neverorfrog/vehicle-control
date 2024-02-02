import time
from matplotlib.backend_bases import FigureManagerBase
from controller.controller import Controller
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from itertools import count, cycle
from environment.trajectory import Trajectory
from model.robot import Robot

class TrajectoryTrackingSimulation():   
    def __init__(self, name: str, robot: Robot, controller: Controller, trajectory: Trajectory):
        self.name = name
        self.robot = robot
        self.controller = controller
        self.trajectory = trajectory
        
    def run(self, N: int = None, animate: bool = True):
        
        # Logging containers
        state_traj = [self.robot.state] # state trajectory (logging)
        action_traj = [] # input trajectory (logging)
        ref_traj = []
        error_traj = []
        elapsed = [] # elapsed times
        
        # Initializing simulation
        state = state_traj[0] 
        counter = count(start=0)
        steps = N if N is not None else np.inf
        
        # Starting Simulation
        for n in cycle(counter):
            if n > steps: break
            
            # computing control signal
            start = time.time()
            action, ref, error = self.controller.command(self.robot,self.trajectory)
            elapsed_time = time.time() - start
            
            # applying control signal
            state = self.robot.drive(action)
            
            # logging
            state_traj.append(state)
            action_traj.append(action)
            ref_traj.append(ref)
            error_traj.append(error)
            elapsed.append(elapsed_time)
         
        print("FINISHED")   
        if animate:
            self.animate(state_traj, action_traj, ref_traj, error_traj, elapsed)   
        
    
    def animate(self, state_traj: list, input_traj: list, ref_traj: list, error_traj: list, elapsed: list):
        assert isinstance(state_traj,list), "State trajectory has to be a list"
        assert isinstance(input_traj,list), "Input trajectory has to be a list"
        
        # simulation params
        N = len(input_traj)
        input_traj = np.array(input_traj)
        state_traj = np.array(state_traj)
        ref_traj = np.array(ref_traj)
        error_traj = np.array(error_traj)
        x_traj = state_traj[:,self.robot.state.index('x')]
        y_traj = state_traj[:,self.robot.state.index('y')]
        x_ref_traj = ref_traj[:,self.robot.state.index('x')]
        y_ref_traj = ref_traj[:,self.robot.state.index('y')]
        time = np.array(state_traj)[:,self.robot.state.index('t')] # for ascissa in side plots
        
        # figure params
        grid = GridSpec(3, 2, width_ratios=[2, 1])
        ax_large = plt.subplot(grid[:, 0])
        ax_small1 = plt.subplot(grid[0, 1])
        ax_small2 = plt.subplot(grid[1, 1])
        ax_small3 = plt.subplot(grid[2, 1])
        state_max = max(state_traj.min(), state_traj.max(), key=abs) # for axis limits
        input_max = max(input_traj.min(), input_traj.max(), key=abs) # for axis limits
        error_max = max(error_traj.min(), error_traj.max(), key=abs) # for axis limits
        pos_max = max(state_traj[:,:2].min(), state_traj[:,:2].max(), ref_traj.max(), ref_traj.min(), key=abs) # for axis limits
        
        # fig titles
        elapsed_time = plt.gcf().text(0.4, 0.9, 'Average computation time', fontsize=16, ha='center', va='center')
        
        def update(i):
            if np.mod(i,5) == 0 and i > 0:
                elapsed_time.set_text(f"Average computation time: {np.mean(elapsed[i-5:i])*1000:.2f} ms")
                
            start = i - 20 if (i - 20 >= 0) else 0
            
            ax_large.cla()
            ax_large.set_aspect('equal')
            self.robot.plot(ax_large, state_traj[i,:])
            ax_large.plot(x_traj[start:i+1],y_traj[start:i+1],'-',alpha=0.3,color="k",linewidth=4)
            ax_large.plot(x_ref_traj[start:i+1],y_ref_traj[start:i+1],'-',alpha=0.7,color="g",linewidth=4)
            ax_large.axis((-pos_max*1.2, pos_max*1.2, -pos_max*1.2, pos_max*1.2))
            
            ax_small1.cla()
            ax_small1.axis((0, time[-1], -state_max*1.1, state_max*1.1))
            ax_small1.plot(time[:i],state_traj[:i,:-1], '-', alpha=0.7,label=['x','y','psi'])
            ax_small1.legend()

            ax_small2.cla()
            ax_small2.axis((0, time[-1], -input_max*1.1, input_max*1.1))
            ax_small2.plot(time[:i],input_traj[:i, :], '-', alpha=0.7,label=['v','w'])
            ax_small2.legend()
            
            ax_small3.cla()
            ax_small3.axis((0, time[-1], -error_max*1.1, error_max*1.1))
            ax_small3.plot(time[:i],error_traj[:i, :], '-', alpha=0.7,label=['e_x','e_y'])
            ax_small3.legend()

        animation = FuncAnimation(
            fig=plt.gcf(), func=update, 
            frames=N, interval=0, 
            repeat=False, repeat_delay=5000
        )
        plt.ioff() #interactive mode off
        animation.save(f"simulation/videos/{self.name}.gif",writer='pillow',fps=20, dpi=180)
        plt.ion() #interactive mode on
        fig_manager: FigureManagerBase = plt.get_current_fig_manager()
        fig_manager.window.showMaximized() 
        plt.show(block=True) 
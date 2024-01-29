import time
from matplotlib.backend_bases import FigureManagerBase
from controller.controller import Controller
import numpy as np
from model.kinematic_car import KinematicCar
from utils.utils import wrap
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from itertools import count, cycle
from casadi import cos, sin, tan
import casadi as ca

class RacingSimulation():   
    def __init__(self, name: str, car: KinematicCar, controller: Controller):
        self.name = name
        self.car = car
        self.controller = controller
        
    def run(self, N: int = None, animate: bool = True):
        
        # Logging containers
        state_traj = [self.car.state] # state trajectory (logging)
        action_traj = [] # input trajectory (logging)
        elapsed = [] # elapsed times
        preds = [] # state predictions for each horizon
        
        # Initializing simulation
        state = state_traj[0] 
        counter = count(start=0)
        steps = N if N is not None else np.inf
        
        # Starting Simulation
        for n in cycle(counter):
            print(f"state: {state}")
            print(f"N: {n}")
            if state.s > self.car.track.length or n >= steps: break
            
            # computing control signal
            curvature = self.car.track.get_curvature(state.s)
            start = time.time()
            action, state_prediction = self.controller.command(state, curvature)
            elapsed_time = time.time() - start
            
            # applying control signal
            state = self.car.drive(action)
            
            # logging
            state_traj.append(state)
            action_traj.append(action)
            elapsed.append(elapsed_time)
            preds.append(np.array([self.car.rel2glob(state_prediction[:,i]) for i in range(self.controller.N)]).squeeze()) # converting prediction to global coordinates
         
        print("FINISHED")   
        if animate:
            self.animate(state_traj, action_traj, preds, elapsed)   
        
    
    def animate(self, state_traj: list, input_traj: list, preds: list, elapsed: list):
        assert isinstance(state_traj,list), "State trajectory has to be a list"
        assert isinstance(input_traj,list), "Input trajectory has to be a list"
        
        # simulation params
        N = len(input_traj)
        v_delta = np.array(state_traj)[:,0:2] # taking just v,psi,delta
        a_w = np.array(input_traj)
        x_traj = []
        y_traj = []
        
        # figure params
        grid = GridSpec(2, 2, width_ratios=[2, 1])
        ax_large = plt.subplot(grid[:, 0])
        ax_small1 = plt.subplot(grid[0, 1])
        ax_small2 = plt.subplot(grid[1, 1])
        state_max = max(v_delta.min(), v_delta.max(), key=abs) # for axis limits
        input_max = max(a_w.min(), a_w.max(), key=abs) # for axis limits
        
        # fig titles
        lap_time = plt.gcf().text(0.4, 0.95, 'Laptime', fontsize=16, ha='center', va='center')
        elapsed_time = plt.gcf().text(0.4, 0.9, 'Mean computation time', fontsize=16, ha='center', va='center')
        
        def update(i):
            state = state_traj[i]
            
            lap_time.set_text(f"Lap time: {state.t:.2f} s") 
            
            if np.mod(i,5) == 0 and i > 0:
                elapsed_time.set_text(f"Average computation time: {np.mean(elapsed[i-5:i])*1000:.2f} ms")
            
            ax_large.cla()
            ax_large.set_aspect('equal')
            x,y = self.car.plot(ax_large, state)
            x_traj.append(x)
            y_traj.append(y)
            ax_large.plot(x_traj[:i+1],y_traj[:i+1],'-',alpha=0.7,color="k",linewidth=4)
            
            # Plot track
            if self.car.track is not None:
                self.car.track.plot(ax_large)
                
            # Plot state predictions of MPC
            if preds is not None and i <= N:
                ax_large.plot(preds[i][:,0], preds[i][:,1],"g",alpha=0.85,linewidth=4)  
            
            ax_small1.cla()
            ax_small1.axis((0, N, -state_max*1.1, state_max*1.1))
            ax_small1.plot(v_delta[:i, :], '-', alpha=0.7,label=['v',r'$\delta$'])
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
        plt.ioff() #interactive mode off
        animation.save(f"simulation/videos/{self.name}.gif",writer='pillow',fps=20, dpi=180)
        plt.ion() #interactive mode on
        fig_manager: FigureManagerBase = plt.get_current_fig_manager()
        fig_manager.window.showMaximized() 
        plt.show(block=True) 
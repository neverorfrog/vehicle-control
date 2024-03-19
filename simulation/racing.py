import time
from matplotlib.backend_bases import FigureManagerBase
from controllers.controller import Controller
import numpy as np
from controllers.mpc.dynamic_mpc import DynamicMPC
from models.racing_car import RacingCar
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from itertools import count, cycle
from models.dynamic_car import DynamicCarInput
from utils.fancy_vector import FancyVector

class RacingSimulation():   
    def __init__(self, name: str, car: RacingCar, point_mass: RacingCar, controller: Controller):
        self.name = name
        self.car = car
        self.point_mass = point_mass
        self.controller = controller
        
    def run(self, N: int = None, animate: bool = True):
        
        # Logging containers
        state_traj = [self.car.state] # state trajectory (logging)
        action_traj = [] # input trajectory (logging)
        elapsed = [] # elapsed times
        preds = [] # state predictions for each horizon
        
        # Initializing simulation
        state: FancyVector = state_traj[0] 
        counter = count(start=0)
        steps = N if N is not None else np.inf
        
        # Starting Simulation
        for n in cycle(counter):
            if state.s > self.car.track.length-0.3 or n > steps: break
            
            # ----------- Computing control signal ------
            start = time.time()
            state_prediction = None
            try:
                action = self.controller.command(state)
            except Exception as e:
                print(e)
                break
            elapsed_time = time.time() - start
            
            # ----------- Applying control signal --------
            state = self.car.drive(action)
            state_prediction = self.controller.state_prediction
            if self.point_mass is not None and self.controller.M > 0:
                state_pm_prediction = self.controller.state_pm_prediction

            # ------------- DEBUG PRINTS -----------------
            # print("------------------------------------------------------------------------------")
            print(f"N: {n}")
            print(f"STATE: {state}")
            print(f"ACTION: {action}")
            print(f"FINAL CURVATURE: {self.car.track.get_curvature(state_prediction[state.index('s'),-1])}")
            print(f"ELAPSED TIME: {elapsed_time}")
            self.car.print(state,action)
            print("------------------------------------------------------------------------------")
            print(f"\n")
            
            # ----------- Logging -------------------------
            state_traj.append(state)
            action_traj.append(action)
            elapsed.append(elapsed_time)
            try:
                preds_car = [self.car.rel2glob(state_prediction[:,i]) for i in range(self.controller.N)]
                if self.point_mass is not None and self.controller.M > 0:
                    preds_pm = [self.point_mass.rel2glob(state_pm_prediction[:,i]) for i in range(self.controller.M)]
                else:
                    preds_pm = []
                preds.append(np.array(preds_car + preds_pm).squeeze())
            except:
                preds = None
        print("FINISHED")   
        if animate:
            # plt.style.use('dark_background')
            self.animate(state_traj, action_traj, preds, elapsed)   
        
    
    def animate(self, state_traj: list, input_traj: list, preds: list, elapsed: list):
        assert isinstance(state_traj,list), "State trajectory has to be a list"
        assert isinstance(input_traj,list), "Input trajectory has to be a list"
        
        # simulation params
        N = len(input_traj)
        ey_index = self.car.state.index('ey')
        error = np.array(state_traj)[:,ey_index:ey_index+2] # taking just ey and epsi
        v = np.array(state_traj)[:,0] # taking just velocity
        s = np.array(state_traj)[:,self.car.state.index('s')] # for ascissa in side plots
        input = np.array(input_traj)
        x_traj = []
        y_traj = []
        
        # figure params
        grid = GridSpec(4, 2, width_ratios=[3, 1])
        plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, hspace=0.3, wspace=0.1)
        ax_large = plt.subplot(grid[:, 0])
        ax_small1 = plt.subplot(grid[0, 1])
        ax_small2 = plt.subplot(grid[1, 1])
        ax_small3 = plt.subplot(grid[2, 1])
        ax_small4 = plt.subplot(grid[3, 1])
        state_max = max(v.min(), v.max(), key=abs) # for axis limits
        error_max = max(error.min(), error.max(), key=abs) # for axis limits
        input_max = max(input.min(), input.max(), key=abs) # for axis limits
        input_labels = self.car.create_input().labels
        error_labels = [r'$e_y$',r'$e_\psi$']
        
        # fig titles
        lap_time = plt.gcf().text(0.4, 0.95, 'Laptime', fontsize=16, ha='center', va='center')
        elapsed_time = plt.gcf().text(0.4, 0.9, 'Mean computation time', fontsize=16, ha='center', va='center')
        
        def update(i):
            state = state_traj[i]
            
            lap_time.set_text(f"Lap time: {state.t:.2f} s | Iteration n.{i}") 
            
            if np.mod(i,5) == 0 and i > 0:
                elapsed_time.set_text(f"Average computation time: {np.mean(elapsed[:i])*1000:.2f} ms")
            
            ax_large.cla()
            ax_large.set_aspect('equal')
            x,y = self.car.plot(ax_large, state)
            x_traj.append(x)
            y_traj.append(y)
            ax_large.plot(x_traj[:i+1],y_traj[:i+1],'-',alpha=0.7,color="k",linewidth=2)
            
            # Plot track
            if self.car.track is not None:
                self.car.track.plot(ax_large)
                
            # Plot state predictions of MPC
            if preds is not None and i <= N:
                ax_large.plot(preds[i][:,0], preds[i][:,1],'g',alpha=0.5,linewidth=3)  
            
            ax_small1.cla()
            ax_small1.axis((s[0], s[-1], -state_max*1.1, state_max*1.1))
            ax_small1.plot(s[:i],v[:i], '-', alpha=0.7,label='v',color='r')
            ax_small1.legend()
                        
            ax_small2.cla()
            ax_small2.axis((s[0], s[-1], -error_max*1.1, error_max*1.1))
            ax_small2.plot(s[:i],error[:i, :], '-', alpha=0.7,label=error_labels)
            ax_small2.legend()

            ax_small3.cla()
            ax_small3.axis((s[0], s[-1], -input_max*1.1, input_max*1.1))
            ax_small3.plot(s[:i],input[:i, 0], '-', alpha=0.7,label=input_labels[0], color='g')
            ax_small3.legend()
            
            ax_small4.cla()
            ax_small4.axis((s[0], s[-1], -0.6, 0.6))
            ax_small4.plot(s[:i],input[:i, 1], '-', alpha=0.7,label=input_labels[1], color='c')
            ax_small4.legend()

        animation = FuncAnimation(
            fig=plt.gcf(), func=update, 
            frames=N, interval=0, 
            repeat=False, repeat_delay=5000
        )
        fig_manager: FigureManagerBase = plt.get_current_fig_manager()
        fig_manager.window.showMaximized() 
        plt.show(block=True) 
        plt.ioff() #interactive mode off
        # animation.save(f"simulation/videos/{self.name}.gif",writer='pillow',fps=20, dpi=200)
        plt.ion() #interactive mode on
        # print("ANIMATION SAVED")
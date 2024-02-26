import time
from matplotlib.backend_bases import FigureManagerBase
from controller.controller import Controller
import numpy as np
from model.racing_car import RacingCar
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from itertools import count, cycle
from model.dynamic_car import DynamicCarInput

class RacingSimulation():   
    def __init__(self, name: str, car: RacingCar, controller: Controller):
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
            if state.s > self.car.track.length or n > steps: break
            
            # computing control signal
            start = time.time()
            state_prediction = None
            """try:action, state_prediction, action_prediction, curvature_prediction = self.controller.command(state)
            except Exception as e:
                print(e)
                break
            """
            elapsed_time = time.time() - start

            action = DynamicCarInput(1, 1)
            if n>=25: 
                action = DynamicCarInput(1, -1)
                
            ##DEBUG PRINTS
            if n <= steps:
                print(f"\n\nN: {n}")
                conv_state = self.car.rel2glob(state)
                print(f"STATE: {state}")
                # print(f"MEASURED POSE: {conv_state[0].item():.3f}, {conv_state[1]:.3f}, {conv_state[2]:.3f}")
                # print(f"REFERENCE POSE: {self.car.track.x(state.s).full().item():.3f}, {self.car.track.y(state.s).full().item():.3f}, {self.car.track.get_orientation(state.s).full().item():.3f}")
                print(f"ACTION: {action}")
                # print(f"FRONT FORCE: {self.car.Fx_f(action[0])}")
                # print(f"REAR FORCE: {self.car.Fx_r(action[0])}")
                # print(f"V PREDICTION: {state_prediction[self.car.state.index('Ux'),:]}")
                # print(f"EY PREDICTION: {state_prediction[self.car.state.index('ey'),:]}")
                #print(f"EPSI PREDICTION: {state_prediction[self.car.state.index('epsi'),:]}")
                # print(f"DELTA PREDICTION: {state_prediction[self.car.state.index('delta'),:]}")
                # print(f"TIME PREDICTION: {state_prediction[self.car.state.index('t'),:]}")
                # print(f"S PREDICTION: {state_prediction[self.car.state.index('s'),:]}")
                # print(f"ACCELERATION PREDICTION: {action_prediction[0,:]}")
                #print(f"OMEGA PREDICTION: {action_prediction[1,:]}")
                # print(f"CURVATURE PREDICTION: {curvature_prediction}")
                print(f"ELAPSED TIME: {elapsed_time}")
                print("")
            ##DEBUG PRINTS
            
            # applying control signal
            state = self.car.drive(action)
            self.car.print(state.values, action.values)
            
            # logging
            state_traj.append(state)
            action_traj.append(action)
            elapsed.append(elapsed_time)
            try:preds.append(np.array([self.car.rel2glob(state_prediction[:,i]) for i in range(self.controller.N)]).squeeze()) # converting prediction to global coordinates
            except: preds = None
        print("FINISHED")   
        if animate:
            self.animate(state_traj, action_traj, preds, elapsed)   
        
    
    def animate(self, state_traj: list, input_traj: list, preds: list, elapsed: list):
        assert isinstance(state_traj,list), "State trajectory has to be a list"
        assert isinstance(input_traj,list), "Input trajectory has to be a list"
        
        # simulation params
        N = len(input_traj)
        ey_index = self.car.state.index('ey')
        error = np.array(state_traj)[:,ey_index:ey_index+2] # taking just ey and epsi
        v = np.array(state_traj)[:,0] # taking just v
        delta = np.array(state_traj)[:,self.car.state.index('delta')] # taking just delta
        s = np.array(state_traj)[:,self.car.state.index('s')] # for ascissa in side plots
        input = np.array(input_traj)
        x_traj = []
        y_traj = []
        
        # figure params
        grid = GridSpec(3, 2, width_ratios=[3, 1])
        ax_large = plt.subplot(grid[:, 0])
        ax_small1 = plt.subplot(grid[0, 1])
        ax_small2 = plt.subplot(grid[1, 1])
        ax_small3 = plt.subplot(grid[2, 1])
        state_max = max(v.min(), v.max(), delta.min(), delta.max(), key=abs) # for axis limits
        error_max = max(error.min(), error.max(), key=abs) # for axis limits
        input_max = max(input.min(), input.max(), key=abs) # for axis limits
        
        # fig titles
        lap_time = plt.gcf().text(0.4, 0.95, 'Laptime', fontsize=16, ha='center', va='center')
        elapsed_time = plt.gcf().text(0.4, 0.9, 'Mean computation time', fontsize=16, ha='center', va='center')
        
        def update(i):
            state = state_traj[i]
            
            lap_time.set_text(f"Lap time: {state.t:.2f} s | Iteration n.{i}") 
            
            if np.mod(i,5) == 0 and i > 0:
                elapsed_time.set_text(f"Average computation time: {np.mean(elapsed[i-5:i])*1000:.2f} ms")
            
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
            ax_small1.plot(s[:i],v[:i], '-', alpha=0.7,label='v')
            ax_small1.plot(s[:i],delta[:i], '-', alpha=0.7,label=r'$\delta$')
            ax_small1.legend()
                        
            ax_small2.cla()
            ax_small2.axis((s[0], s[-1], -error_max*1.1, error_max*1.1))
            ax_small2.plot(s[:i],error[:i, :], '-', alpha=0.7,label=[r'$e_y$',r'$e_\psi$'])
            ax_small2.legend()

            ax_small3.cla()
            ax_small3.axis((s[0], s[-1], -input_max*1.1, input_max*1.1))
            ax_small3.plot(s[:i],input[:i, :], '-', alpha=0.7,label=['a','w'])
            ax_small3.legend()

        animation = FuncAnimation(
            fig=plt.gcf(), func=update, 
            frames=N, interval=0, 
            repeat=False, repeat_delay=5000
        )
        fig_manager: FigureManagerBase = plt.get_current_fig_manager()
        fig_manager.window.showMaximized() 
        plt.show(block=True) 
        plt.ioff() #interactive mode off
        animation.save(f"simulation/videos/{self.name}.gif",writer='pillow',fps=20, dpi=200)
        plt.ion() #interactive mode on
        print("ANIMATION SAVED")
import os
import time
from typing import List
from omegaconf import OmegaConf
from controllers.controller import Controller
import numpy as np
from controllers.mpc.kinematic_mpc import KinematicMPC
from environment.track import Track
from models.dynamic_car import DynamicCarState
from models.kinematic_car import KinematicCar
from models.racing_car import RacingCar
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

class RacingSimulation():
    """
    Class for running a simulation of racing cars

    This class runs a simulation of racing cars on a track. It uses
    a list of models, controllers, and a track to generate the
    simulation. It also generates animations of the simulation
    using matplotlib.
    """
    def __init__(self, names: List[str], cars: List[RacingCar], controllers: List[Controller], track: Track):
        self.names = names
        self.cars = cars
        self.controllers = controllers
        self.track = track
        
    def run(self, N: int = None):
        """
        Run the simulation for a specified number of steps

        Parameters
        ----------
        N: int
            Number of steps to run the simulation for. If None, the simulation will run indefinitely

        Returns
        -------
        state_traj, action_traj, preds, elapsed
            Trajectories of the state, action, and state predictions for each car as a dictionary.
            The "elapsed" dictionary contains the elapsed time for each car's controller.
        """
        # Logging containers
        state_traj = {name: [car.state] for name,car in zip(self.names,self.cars)} # state trajectory (logging)
        action_traj = {name: [] for name in self.names} # action trajectory (logging)
        elapsed = {name: [] for name in self.names} # elapsed times
        preds = {name: [] for name in self.names} # state predictions for each horizon
        
        # Initializing simulation
        n = 0
        steps = N if N is not None else np.inf
        track_length = self.track.length
        
        # Starting Simulation
        while(True):
            if n > steps: break
            for name,car,controller in zip(self.names,self.cars,self.controllers):
                state = car.state
                if state.s > track_length-1: n = steps + 1
                
                # ----------- Applying control signal --------------------------
                try:
                    start = time.time()
                    action, state = controller.command(state)
                    elapsed_time = time.time() - start
                except Exception as e:
                    print(e)
                    n = steps + 1
                    break
            
                # ----------- Saving trajectories --------------------------------
                state_traj[name].append(state)
                action_traj[name].append(action)
                elapsed[name].append(elapsed_time)
                preds[name].append(controller.get_state_prediction())
                
                # ----------- Logging prints -------------------------------------
                print("------------------------------------------------------------------------------")
                print(f"N: {n}")
                print(f"STATE: {state}")
                print(f"ACTION: {action}")
                print(f"AVERAGE ELAPSED TIME: {np.mean(elapsed[name]):.3f}")
                print(f"MEDIAN ELAPSED TIME: {np.median(elapsed[name]):.3f}")
                car.print(state,action)
                print("------------------------------------------------------------------------------")
                print(f"\n")
            n += 1
        return state_traj, action_traj, preds, elapsed
      
    def save(self, state_traj: dict, action_traj: dict, preds: dict, elapsed: dict):
        assert isinstance(state_traj,dict), "State trajectory has to be a dict"
        assert isinstance(action_traj,dict), "Action trajectory has to be a dict"
        for name, controller in zip(self.names, self.controllers):
            path = f"simulation/data/{self.track.name}/{name}"
            os.makedirs(path, exist_ok=True)
            np.save(f"{path}/state_traj.npy", state_traj[name])
            np.save(f"{path}/action_traj.npy", action_traj[name])
            np.save(f"{path}/preds.npy", preds[name])
            np.save(f"{path}/elapsed.npy", elapsed[name])  
            OmegaConf.save(config=controller.config, f=f"simulation/data/{self.track.name}/{name}/config.yaml")
    
    def load(self):
        pass  
    
    def animate(self, state_traj: dict, action_traj: dict, preds: dict, elapsed: dict):
        """
        Plots the state, action, and predicted state trajectories for each car.

        Parameters
        ----------
        state_traj: dict
            Dictionary of state trajectories for each car
        action_traj: dict
            Dictionary of action trajectories for each car
        preds: dict
            Dictionary of state predictions for each horizon for each car
        elapsed: dict
            Dictionary of elapsed times for each car

        Returns
        -------
        animation: FuncAnimation
            The animation object
        """
        assert isinstance(state_traj,dict), "State trajectory has to be a dict"
        assert isinstance(action_traj,dict), "Action trajectory has to be a dict"
        
        # simulation params
        ey_index = self.cars[0].state.index('ey')
        
        error = []; v = []; s = []; delta = []; actions = []; states = []; x_traj = []; y_traj = []
        for name,car in zip(self.names,self.cars):
            traj = np.array(state_traj[name])
            error.append(traj[:,ey_index]) # taking just ey
            v.append(traj[:,0]) # taking just velocity
            delta.append(traj[:,car.state.index('delta')])
            s.append(traj[:,car.state.index('s')]) # for ascissa in side plots
            actions.append(np.array(action_traj[name]))
            states.append(traj)
            x_traj.append([])
            y_traj.append([])
            
        # figure params
        grid = GridSpec(4, 2, width_ratios=[3, 1])
        plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.85, hspace=0.3, wspace=0.1)
        ax_large = plt.subplot(grid[:, 0])
        ax_small1 = plt.subplot(grid[0, 1])
        ax_small2 = plt.subplot(grid[1, 1]) 
        ax_small3 = plt.subplot(grid[2, 1])
        ax_small4 = plt.subplot(grid[3, 1]) 
        lap_time = plt.gcf().text(0.4, 0.95, 'Laptime', fontsize=16, ha='center', va='center')
        elapsed_time = plt.gcf().text(0.4, 0.9, 'Average time', fontsize=16, ha='center', va='center')
        mean_speed = plt.gcf().text(0.4, 0.85, 'Mean speed', fontsize=16, ha='center', va='center')
        
        colors = ['g','y','r']
        
        def clear():
            ax_large.cla()
            ax_large.set_aspect('equal')
            ax_small1.cla()
            ax_small2.cla()
            ax_small3.cla()
            ax_small4.cla()
            ax_small1.axis((s[0][0], s[0][-1], 0, 20))
            ax_small1.set_title(r'$v$', loc='center')
            ax_small2.axis((s[0][0], s[0][-1], -0.5, 0.5))
            ax_small2.set_title(r'$\delta$', loc='center')
            ax_small3.axis((s[0][0], s[0][-1], -4, 4))
            ax_small3.set_title(r'$e_y$', loc='center')
            ax_small4.axis((s[0][0], s[0][-1], -7000, 7000))
            ax_small4.set_title(r'$F_x$', loc='center')
            return ax_large, ax_small1, ax_small2, ax_small3, ax_small4
            
            
        def update(frame):
            clear()
            #Plot Track
            self.track.plot(ax_large)
            if self.controllers[0].config.obstacles:
                for obs in self.track.obstacles:
                    obs.plot(ax_large)
            # Plot text
            lap_time.set_text(f"Iteration n.{frame} | Laptime {state_traj[self.names[0]][frame][-1]:.2f} s")
            if np.mod(frame,5) == 0 and frame > 0:
                time = f"Median time"
                speed = f"Mean Speed"
                for j in range(len(self.names)):
                    time += f" | {self.names[j]} = {np.median(elapsed[self.names[j]][:frame])*1000:.2f} ms"
                    speed += f" | {self.names[j]} = {np.mean(v[j][:frame]):.2f} m/s"
                    elapsed_time.set_text(time)
                    mean_speed.set_text(speed)  
                      
            for j in range(len(self.names)):
                state = state_traj[self.names[j]][frame]
            
                # Plot car
                x,y = self.cars[j].plot(ax_large, state, colors[j])
                x_traj[j].append(x)
                y_traj[j].append(y)
                ax_large.plot(x_traj[j][:frame],y_traj[j][:frame],'-',alpha=0.8,color=colors[j],linewidth=3)
            
                # Plot state predictions of MPC
                if preds is not None:
                    ax_large.plot(preds[self.names[j]][frame][:,0], preds[self.names[j]][frame][:,1],f'{colors[j]}o',alpha=0.5,linewidth=3)  
            
                # Plot state and actions
                ax_small1.plot(s[j][:frame],v[j][:frame], '-', alpha=0.7, label=self.names[j], color = colors[j]); ax_small1.legend()
                ax_small2.plot(s[j][:frame],delta[j][:frame], '-', alpha=0.7, label=self.names[j], color = colors[j]); ax_small2.legend()
                ax_small3.plot(s[j][:frame],error[j][:frame], '-', alpha=0.7, label=self.names[j], color = colors[j]); ax_small3.legend()
                ax_small4.plot(s[j][:frame],actions[j][:frame, 0], '-', alpha=0.7, label=self.names[j], color = colors[j]); ax_small4.legend()
            return ax_large, ax_small1, ax_small2, ax_small3, ax_small4

        animation = FuncAnimation(fig=plt.gcf(), func=update, frames=len(action_traj[self.names[0]]), interval=1, repeat=False)
        return animation
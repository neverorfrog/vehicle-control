import os
import time
from typing import List, Union
from matplotlib.animation import FuncAnimation
from matplotlib.backend_bases import FigureManagerBase
from omegaconf import OmegaConf
from controllers.controller import Controller
import numpy as np
from environment.track import Track
from models.racing_car import RacingCar
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
        self.colors = ['g','y','r']
        self.init_containers()
        self.init_simulation()
    
        
    def init_containers(self):
        # Logging containers
        self.state_traj = {name: [car.state] for name,car in zip(self.names,self.cars)} # state trajectory (logging)
        self.action_traj = {name: [car.create_action()] for name,car in zip(self.names,self.cars)} # action trajectory (logging)
        self.elapsed = {name: [] for name in self.names} # elapsed times
        self.preds = {name: [] for name in self.names} # state predictions for each horizon
        self.x_traj = [[] for _ in self.names]
        self.y_traj = [[] for _ in self.names]
        
            
    def init_simulation(self):
        # Grid for subplots
        grid = GridSpec(4, 2, width_ratios=[3, 1])
        plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.8, hspace=0.3, wspace=0.1)
        
        # Big axis initialization (one for the track and one for the car)
        self.ax_track = plt.subplot(grid[:, 0])
        self.ax_track.set_aspect('equal')
        self.ax_car = self.ax_track.twinx()
        self.track.plot(self.ax_track)
        if self.controllers[0].config.obstacles:
            for obs in self.track.obstacles:
                obs.plot(self.ax_track)
                
        # Small axes initialization (for plots on s axis)
        self.ax_small1 = plt.subplot(grid[0, 1]); self.ax_small1.axis((0, self.track.length, 0, 22))
        self.ax_small1.set_ylabel(r'$v \rightarrow \frac{m}{s}$', fontsize=16, labelpad=25, rotation=360); self.ax_small1.yaxis.set_label_position('right')
        
        self.ax_small2 = plt.subplot(grid[1, 1]); self.ax_small2.axis((0, self.track.length, -0.5, 0.5))
        self.ax_small2.set_ylabel(r'$\delta \rightarrow rad$', fontsize=16, labelpad=30, rotation=360); self.ax_small2.yaxis.set_label_position('right')
        
        self.ax_small3 = plt.subplot(grid[3, 1]); self.ax_small3.axis((0, self.track.length, -0.5, 0.5))
        self.ax_small3.set_ylabel(r'$\omega \rightarrow \frac{rad}{s}$', fontsize=16, labelpad=30, rotation=360); self.ax_small3.yaxis.set_label_position('right')
        
        self.ax_small4 = plt.subplot(grid[2, 1]); self.ax_small4.axis((0, self.track.length, -7000, 7000))
        self.ax_small4.set_ylabel(r'$F_x \rightarrow N$', fontsize=16, labelpad=25, rotation=360); self.ax_small4.yaxis.set_label_position('right')
        
        
        # Text boxes
        self.lap_time = plt.gcf().text(0.4, 0.95, 'Laptime', fontsize=16, ha='center', va='center')
        self.elapsed_time = plt.gcf().text(0.4, 0.9, 'Average time', fontsize=16, ha='center', va='center')
        self.mean_speed = plt.gcf().text(0.4, 0.85, 'Mean speed', fontsize=16, ha='center', va='center')
        
        # Figure initialization
        fig_manager: FigureManagerBase = plt.get_current_fig_manager()
        self.animation = FuncAnimation(plt.gcf(), self.update, interval=1, cache_frame_data=False)
        fig_manager.window.showMaximized()
        plt.show()

            
    def update(self, n):        
        for name,car,controller in zip(self.names,self.cars,self.controllers):
            if car.state.s > self.track.length-0.1: 
                return
            
            start = time.time()
            action, state = self.step(controller, car)
            elapsed_time = time.time() - start
            self.debug_print(n, car, action, state)
        
            self.state_traj[name].append(state)
            self.action_traj[name].append(action)
            self.elapsed[name].append(elapsed_time)
            self.preds[name].append(controller.get_state_prediction()) # each state prediction is an array of shape [horizon,3]
            
        self.plot(n)
    
    def plot(self,n):
        # Plot text
        self.lap_time.set_text(f"Iteration n.{n} | Laptime {self.state_traj[self.names[0]][n][-1]:.2f} s")
        if np.mod(n,5) == 0 and n > 0:
            time = f"Median time"
            speed = f"Mean Speed"
            for name in self.names:
                v = np.array(self.state_traj[name])[:,0]
                time += f" | {name} = {np.median(self.elapsed[name])*1000:.2f} ms"
                speed += f" | {name} = {np.mean(v):.2f} m/s"
            self.elapsed_time.set_text(time)
            self.mean_speed.set_text(speed) 
        
        
        # Cycle cars     
        self.ax_car.cla()
        self.ax_car.set_aspect('equal')
        self.ax_car.set_ylim([self.ax_track.get_ylim()[0], self.ax_track.get_ylim()[1]])
        for j in range(len(self.names)):
            # Extracting data
            name = self.names[j]
            car = self.cars[j]
            state = self.state_traj[name][n]
            state_traj = np.array(self.state_traj[name])
            action_traj = np.array(self.action_traj[name])
            s = state_traj[:,car.state.index('s')]
            delta = state_traj[:,car.state.index('delta')]
            v = state_traj[:,0]
            Fx = action_traj[:,0]
            w = action_traj[:,1]
            
            # Plot car
            x,y = self.cars[j].plot(self.ax_car, state, self.colors[j])
            self.x_traj[j].append(x)
            self.y_traj[j].append(y)
            self.ax_car.plot(self.x_traj[j],self.y_traj[j],'-',alpha=0.8,color=self.colors[j],linewidth=3)
            
            # Plot state predictions of MPC
            self.ax_car.plot(self.preds[name][n][:,0], self.preds[name][n][:,1],f'{self.colors[j]}o',alpha=0.5,linewidth=3) 
            
            # Plot state and actions
            self.ax_small1.plot(s[-2:],v[-2:], '-',markersize=2, alpha=0.7, label=self.names[j], color = self.colors[j])
            self.ax_small2.plot(s[-2:],delta[-2:],'-',markersize=2, alpha=0.7, label=self.names[j], color = self.colors[j])
            self.ax_small3.plot(s[-2:],w[-2:],'-',markersize=2, alpha=0.7, label=self.names[j], color = self.colors[j])
            self.ax_small4.plot(s[-2:],Fx[-2:],'-',markersize=2, alpha=0.7, label=self.names[j], color = self.colors[j])
            
    
    def step(self, controller, car) -> Union[None, tuple]:
        try:
            action = controller.command(car.state)
            state = car.drive(action)
        except Exception as e:
            print(e)
            return None
        return action, state
    
    
    def debug_print(self, n, car, action, state):
        # ----------- Logging prints -------------------------------------
        print("------------------------------------------------------------------------------")
        print(f"N: {n}")
        # print(f"STATE: {state}")
        # print(f"ACTION: {action}")
        # print(f"AVERAGE ELAPSED TIME: {np.mean(elapsed[name]):.3f}")
        # print(f"MEDIAN ELAPSED TIME: {np.median(elapsed[name]):.3f}")
        car.print(state,action)
        print("------------------------------------------------------------------------------")
        print(f"\n")
        
            
    def save(self):
        for name, controller in zip(self.names, self.controllers):
            path = f"simulation/data/{self.track.name}/{name}"
            os.makedirs(path, exist_ok=True)
            np.save(f"{path}/state_traj.npy", self.state_traj[name])
            np.save(f"{path}/action_traj.npy", self.action_traj[name])
            np.save(f"{path}/preds.npy", self.preds[name])
            np.save(f"{path}/elapsed.npy", self.elapsed[name])  
            OmegaConf.save(config=controller.config, f=f"simulation/data/{self.track.name}/{name}/config.yaml")
    
    def load(self):
        pass
    
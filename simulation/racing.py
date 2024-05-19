import sys
import os
import time
from typing import List, Union
from omegaconf import OmegaConf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure

from controllers.controller import Controller
from environment.track import Track
from models import DynamicCar
from models.racing_car import RacingCar
from simulation.simulation import Simulation

class RacingSimulation(Simulation):
    """
    Class for running a simulation of racing cars

    This class runs a simulation of racing cars on a track. It uses
    a list of models, controllers, and a track to generate the
    simulation. It also generates animations of the simulation
    using matplotlib.
    """
    def __init__(self, name: str, names: List[str], cars: List[RacingCar], controllers: List[Controller], colors: List[str], track: Track, load = False):
        self.names = names
        self.cars = cars
        self.controllers = controllers
        self.colors = colors
        self.track = track
        super().__init__(name,load)
        
    @property
    def state_len(self):
        return len(self.state_traj[self.names[0]])
        
    def init_containers(self):
        # Logging containers
        self.state_traj = {name: [car.state] for name,car in zip(self.names,self.cars)} # state trajectory (logging)
        self.action_traj = {name: [car.create_action()] for name,car in zip(self.names,self.cars)} # action trajectory (logging)
        self.elapsed = {name: [] for name in self.names} # elapsed times
        self.preds = {name: [] for name in self.names} # state predictions for each horizon
        self.x_traj = [[] for _ in self.names]
        self.y_traj = [[] for _ in self.names]
        
            
    def init_animation(self, func: object, fig: Figure = plt.gcf(), frames: int = None):
        # Grid for subplots
        grid = GridSpec(5, 2, width_ratios=[3, 1])
        plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, hspace=0.3, wspace=0.1)
        
        # Big axis initialization (one for the track and one for the car)
        self.ax_track = plt.subplot(grid[:, 0])
        self.ax_track.set_aspect('equal')
        self.ax_car = self.ax_track.twinx()
        self.track.plot(self.ax_track)
        if self.controllers[0].config.obstacles:
            for obs in self.track.obstacles:
                obs.plot(self.ax_track)
                
        # Small axes initialization (for plots on s axis)
        self.ax_small0 = plt.subplot(grid[0, 1]); self.ax_small0.axis((0, self.track.length, 20, 150))
        self.ax_small0.set_ylabel(r'$ms$', fontsize=16, labelpad=25, rotation=360); self.ax_small0.yaxis.set_label_position('right')
        
        self.ax_small1 = plt.subplot(grid[1, 1]); self.ax_small1.axis((0, self.track.length, 0, 22))
        self.ax_small1.set_ylabel(r'$v \rightarrow \frac{m}{s}$', fontsize=16, labelpad=25, rotation=360); self.ax_small1.yaxis.set_label_position('right')
        
        self.ax_small2 = plt.subplot(grid[2, 1]); self.ax_small2.axis((0, self.track.length, -0.5, 0.5))
        self.ax_small2.set_ylabel(r'$\delta \rightarrow rad$', fontsize=16, labelpad=30, rotation=360); self.ax_small2.yaxis.set_label_position('right')
        
        self.ax_small3 = plt.subplot(grid[4, 1]); self.ax_small3.axis((0, self.track.length, -0.5, 0.5))
        self.ax_small3.set_ylabel(r'$\omega \rightarrow \frac{rad}{s}$', fontsize=16, labelpad=30, rotation=360); self.ax_small3.yaxis.set_label_position('right')
        
        self.ax_small4 = plt.subplot(grid[3, 1]); self.ax_small4.axis((0, self.track.length, -7000, 7000))
        self.ax_small4.set_ylabel(r'$F_x \rightarrow N$', fontsize=16, labelpad=25, rotation=360); self.ax_small4.yaxis.set_label_position('right')
        
        # Text boxe
        self.lap_time = fig.text(0.5, 0.97, 'Laptime', fontsize=16, ha='center', va='center')
        
        # Animation initialization
        return FuncAnimation(fig, func, frames, interval=0, cache_frame_data=False, repeat_delay=0)
    
    
    def update(self, n):  
        for car in self.cars:
            if car.state.s > self.track.length-0.1:
                self.logfile.close()
                sys.stdout = sys.__stdout__
                self.save()
                # self.save_animation()
                self.animation.event_source.stop()
                return

        for name,car,controller in zip(self.names,self.cars,self.controllers):
            start = time.time()
            action, state = self.step(controller, car)
            elapsed_time = time.time() - start
            self.debug_print(n, car, self.elapsed[name], action, state)
        
            self.state_traj[name].append(state)
            self.action_traj[name].append(action)
            self.elapsed[name].append(elapsed_time)
            self.preds[name].append(controller.get_state_prediction()) # each state prediction is an array of shape [horizon,3]
        self.plot(n)
    
    def plot(self,n):
        # Plot text
        self.lap_time.set_text(f"Iteration n.{n}     |     Laptime {self.state_traj[self.names[0]][n][-1]:.2f} s")
        
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
            self.ax_car.plot(self.x_traj[j][:n],self.y_traj[j][:n],'-',alpha=0.8,color=self.colors[j],linewidth=2)
            
            # Plot state predictions of MPC
            self.ax_car.plot(self.preds[name][n][:,0],self.preds[name][n][:,1],linestyle='None',color=self.colors[j],marker='o',markerfacecolor=self.colors[j],markersize=4,alpha=0.3) 
            
            # Plot state and actions
            self.ax_small0.plot(s[n],np.mean(self.elapsed[name][:n])*1000, 'o', markersize=0.5,alpha=0.7,linewidth=1,label=self.names[j], color = self.colors[j])
            self.ax_small1.plot(s[n-2:n],v[n-2:n], '-', alpha=0.7,linewidth=1,label=self.names[j], color = self.colors[j])
            self.ax_small2.plot(s[n-2:n],delta[n-2:n],'-', alpha=0.7,linewidth=1,label=self.names[j], color = self.colors[j])
            self.ax_small3.plot(s[n-2:n],w[n-2:n],'-', alpha=0.7,linewidth=1,label=self.names[j], color = self.colors[j])
            self.ax_small4.plot(s[n-2:n],Fx[n-2:n],'-', alpha=0.7,linewidth=1,label=self.names[j], color = self.colors[j])
            
    
    def step(self, controller: Controller, car: DynamicCar) -> Union[None, tuple]:
        try:
            action = controller.command(car.state)
            state = car.drive(action)
        except Exception as e:
            print(e)
            return None
        return action, state
    
    
    def debug_print(self, n, car, elapsed, action, state):
        # ----------- Logging prints -------------------------------------
        print("-----------------------------------------------------------------------------------")
        print(f"N: {n}")
        print(f"STATE: {state}")
        print(f"ACTION: {action}")
        print(f"AVERAGE ELAPSED TIME: {np.mean(elapsed):.3f}")
        print(f"MEDIAN ELAPSED TIME: {np.median(elapsed):.3f}")
        car.print(state,action)
        print("-----------------------------------------------------------------------------------")
        print(f"\n")
             
    def save(self):
        for name, controller in zip(self.names, self.controllers):
            path = f"{self.src_dir}/data/{self.name}/{name}"
            os.makedirs(path, exist_ok=True)
            np.save(f"{path}/state_traj.npy", self.state_traj[name])
            np.save(f"{path}/action_traj.npy", self.action_traj[name])
            np.save(f"{path}/preds.npy", self.preds[name])
            np.save(f"{path}/elapsed.npy", self.elapsed[name])  
            OmegaConf.save(config=controller.config, f=f"{path}/config.yaml")
    
    def load(self):
        for name, controller in zip(self.names, self.controllers):
            path = f"{self.src_dir}/data/{self.name}/{name}"
            self.state_traj[name] = np.load(f"{path}/state_traj.npy")
            self.action_traj[name] = np.load(f"{path}/action_traj.npy")
            self.preds[name] = np.load(f"{path}/preds.npy")
            self.elapsed[name] = np.load(f"{path}/elapsed.npy")
            controller.config = OmegaConf.load(f"{path}/config.yaml")
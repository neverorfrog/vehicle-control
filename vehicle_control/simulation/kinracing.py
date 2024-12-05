import os
import sys
import time
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from omegaconf import OmegaConf

import vehicle_control.controllers as control
import vehicle_control.models as models
import vehicle_control.utils.common_utils as utils
from vehicle_control.controllers.controller import Controller
from vehicle_control.environment.track import Track
from vehicle_control.models import DynamicCar
from vehicle_control.simulation.simulator import Simulator


class KinematicRacingSimulator(Simulator):
    """
    Class for running a simulation of kinematic racing cars

    This class runs a simulation of racing cars on a track. It uses
    a list of models, controllers, and a track to generate the
    simulation. It also generates animations of the simulation
    using matplotlib.
    """

    def __init__(
        self, simconfig: OmegaConf, carconfig: OmegaConf, trackconfig: OmegaConf
    ):
        self.names = simconfig.controller_names

        # track
        track = Track(trackconfig)
        self.track = track

        # cars
        cars = [
            models.KinematicCar(config=carconfig, track=self.track) for _ in self.names
        ]
        for car in cars:
            car.state = models.KinematicCarState(v=0.1, s=1)
        self.cars = cars

        # controllers
        controllerconfigs = [
            OmegaConf.create(utils.load_config("config/controllers/kinematic.yaml"))
            for _ in self.names
        ]
        combriccola = zip(cars, controllerconfigs)
        controllers = [
            control.KinematicMPC(car=car, config=config) for car, config in combriccola
        ]
        self.controllers = controllers
        self.colors = [controller.config.color for controller in controllers]

        super().__init__(simconfig)

    @property
    def name(self):
        return f"{self.config.name}_{self.config.track_name}"

    @property
    def state_len(self):
        return len(self.state_traj[self.names[0]])

    def init_containers(self):
        # Logging containers
        self.state_traj = {
            name: [car.state] for name, car in zip(self.names, self.cars)
        }  # state trajectory (logging)
        self.action_traj = {
            name: [car.create_action()] for name, car in zip(self.names, self.cars)
        }  # action trajectory (logging)
        self.elapsed = {name: [] for name in self.names}  # elapsed times
        self.preds = {
            name: [] for name in self.names
        }  # state predictions for each horizon
        self.x_traj = [[] for _ in self.names]
        self.y_traj = [[] for _ in self.names]

    def summarize(self):
        if not self.loaded:
            return
        print(self.name)
        for name in self.names:
            print("-------------------------")
            print(name)
            print(f"Laptime: {self.state_traj[name][-1,-1]}")
            print(f"Average time:{np.mean(self.elapsed[name])}")
            print(f"Average speed: {np.mean(self.state_traj[name][:,0])}")
            print(f"Mean squared error: {np.mean(np.square(self.state_traj[name][:,5]))}")
            print("-------------------------")

    def init_animation(self, func: object, fig: Figure = plt.gcf(), frames: int = None):
        # Grid for subplots
        grid = GridSpec(5, 2, width_ratios=[3, 1])
        plt.subplots_adjust(
            left=0.05, bottom=0.05, right=0.95, top=0.95, hspace=0.3, wspace=0.1
        )

        # Big axis initialization (one for the track and one for the car)
        self.ax_track = plt.subplot(grid[:, 0])
        self.ax_track.set_aspect("equal")
        self.ax_car: Axes = self.ax_track.twinx()
        self.track.plot(self.ax_track)
        if self.controllers[0].config.obstacles:
            for obs in self.track.obstacles:
                obs.plot(self.ax_track)

        # Small axes initialization (for plots on s axis)
        self.ax_small0 = plt.subplot(grid[0, 1])
        self.ax_small0.axis((0, self.track.length, 0, 30))
        self.ax_small0.set_ylabel(r"$ms$", fontsize=16, labelpad=25, rotation=360)
        self.ax_small0.yaxis.set_label_position("right")

        self.ax_small1 = plt.subplot(grid[1, 1])
        self.ax_small1.axis((0, self.track.length, 0, 50))
        self.ax_small1.set_ylabel(
            r"$v \rightarrow \frac{m}{s}$", fontsize=16, labelpad=25, rotation=360
        )
        self.ax_small1.yaxis.set_label_position("right")

        self.ax_small2 = plt.subplot(grid[2, 1])
        self.ax_small2.axis((0, self.track.length, -0.5, 0.5))
        self.ax_small2.set_ylabel(
            r"$\delta \rightarrow rad$", fontsize=16, labelpad=30, rotation=360
        )
        self.ax_small2.yaxis.set_label_position("right")

        self.ax_small3 = plt.subplot(grid[4, 1])
        self.ax_small3.axis((0, self.track.length, -0.5, 0.5))
        self.ax_small3.set_ylabel(
            r"$\omega \rightarrow \frac{rad}{s}$", fontsize=16, labelpad=30, rotation=360
        )
        self.ax_small3.yaxis.set_label_position("right")

        self.ax_small4 = plt.subplot(grid[3, 1])
        self.ax_small4.axis((0, self.track.length, -20, 20))
        self.ax_small4.set_ylabel(
            r"$a_x \rightarrow N$", fontsize=16, labelpad=25, rotation=360
        )
        self.ax_small4.yaxis.set_label_position("right")

        # Text boxe
        self.lap_time = fig.text(
            0.5, 0.97, "Laptime", fontsize=16, ha="center", va="center"
        )

        # Animation initialization
        return FuncAnimation(
            fig, func, frames, interval=0, cache_frame_data=False, repeat_delay=0
        )

    def update(self, n):
        for car in self.cars:
            if car.state.s > self.track.length - 0.1:
                if self.config.logging:
                    self.logfile.close()
                    sys.stdout = sys.__stdout__
                if self.config.save_data:
                    self.save()
                if self.config.save_gif:
                    self.save_animation()
                self.animation.event_source.stop()
                return

        for name, car, controller in zip(self.names, self.cars, self.controllers):
            start = time.time()
            action, state = self.step(controller, car)
            elapsed_time = time.time() - start
            self.debug_print(n, car, self.elapsed[name], action, state)

            self.state_traj[name].append(state)
            self.action_traj[name].append(action)
            self.elapsed[name].append(elapsed_time)
            self.preds[name].append(
                controller.get_state_prediction()
            )  # each state prediction is an array of shape [horizon,3]
        self.plot(n)

    def plot(self, n):
        # Plot text
        self.lap_time.set_text(
            f"Iteration n.{n}     |     Laptime {self.state_traj[self.names[0]][n][-1]:.2f} s"
        )

        # Cycle cars
        self.ax_car.cla()
        self.ax_car.set_aspect("equal")
        self.ax_car.set_ylim([self.ax_track.get_ylim()[0], self.ax_track.get_ylim()[1]])
        for j in range(len(self.names)):
            # Extracting data
            name = self.names[j]
            car = self.cars[j]
            state = self.state_traj[name][n]
            state_traj = np.array(self.state_traj[name])
            action_traj = np.array(self.action_traj[name])
            s = state_traj[:, car.state.index("s")]
            delta = state_traj[:, car.state.index("delta")]
            v = state_traj[:, 0]
            Fx = action_traj[:, 0]
            w = action_traj[:, 1]

            # Plot car
            x, y = self.cars[j].plot(self.ax_car, state, self.colors[j])
            self.x_traj[j].append(x)
            self.y_traj[j].append(y)
            self.ax_car.plot(
                self.x_traj[j][:n],
                self.y_traj[j][:n],
                "-",
                alpha=0.7,
                color=self.colors[j],
                linewidth=2,
                label=self.names[j],
            )
            self.ax_car.legend(prop={"size": 15})

            # Plot state predictions of MPC
            self.ax_car.plot(
                self.preds[name][n][:, 0],
                self.preds[name][n][:, 1],
                linestyle="None",
                color=self.colors[j],
                marker="o",
                markerfacecolor=self.colors[j],
                markersize=4,
                alpha=0.3,
            )

            # Plot state and actions
            self.ax_small0.plot(
                s[n],
                np.mean(self.elapsed[name][:n]) * 1000,
                "o",
                markersize=0.7,
                alpha=0.7,
                linewidth=1,
                color=self.colors[j],
            )
            self.ax_small1.plot(
                s[n - 2 : n],
                v[n - 2 : n],
                "-",
                alpha=0.7,
                linewidth=1,
                color=self.colors[j],
            )
            self.ax_small2.plot(
                s[n - 2 : n],
                delta[n - 2 : n],
                "-",
                alpha=0.7,
                linewidth=1,
                color=self.colors[j],
            )
            self.ax_small3.plot(
                s[n - 2 : n],
                w[n - 2 : n],
                "-",
                alpha=0.7,
                linewidth=1,
                color=self.colors[j],
            )
            self.ax_small4.plot(
                s[n - 2 : n],
                Fx[n - 2 : n],
                "-",
                alpha=0.7,
                linewidth=1,
                color=self.colors[j],
            )

        if self.config.save_images:
            plt.gcf().savefig(f"{self.images_path}/frame{n}.png", dpi=50)

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
        print(
            "-----------------------------------------------------------------------------------"
        )
        print(f"N: {n}")
        print(f"STATE: {state}")
        print(f"ACTION: {action}")
        print(f"AVERAGE ELAPSED TIME: {np.mean(elapsed):.3f}")
        print(f"MEDIAN ELAPSED TIME: {np.median(elapsed):.3f}")
        car.print(state, action)
        print(
            "-----------------------------------------------------------------------------------"
        )
        print("\n")

    def save(self):
        os.makedirs(self.data_path, exist_ok=True)
        for name, controller in zip(self.names, self.controllers):
            np.save(f"{self.data_path}/{name}_state_traj.npy", self.state_traj[name])
            np.save(f"{self.data_path}/{name}_action_traj.npy", self.action_traj[name])
            np.save(f"{self.data_path}/{name}_preds.npy", self.preds[name])
            np.save(f"{self.data_path}/{name}_elapsed.npy", self.elapsed[name])
            OmegaConf.save(
                config=controller.config, f=f"{self.data_path}/{name}_config.yaml"
            )

    def load(self):
        for name, controller in zip(self.names, self.controllers):
            self.state_traj[name] = np.load(f"{self.data_path}/{name}_state_traj.npy")
            self.action_traj[name] = np.load(f"{self.data_path}/{name}_action_traj.npy")
            self.preds[name] = np.load(f"{self.data_path}/{name}_preds.npy")
            self.elapsed[name] = np.load(f"{self.data_path}/{name}_elapsed.npy")
            controller.config = OmegaConf.load(f"{self.data_path}/{name}_config.yaml")
        self.loaded = True

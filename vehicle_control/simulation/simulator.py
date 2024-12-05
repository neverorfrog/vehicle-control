import os
import sys
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure
from omegaconf import OmegaConf

from vehicle_control.utils.common_utils import project_root


class Simulator(ABC):

    def __init__(self, config: OmegaConf):
        self.init_containers()
        self.config = config
        self.root = project_root()
        self.data_path = f"{self.root}/experiments/data/{self.name}"
        os.makedirs(self.data_path, exist_ok=True)
        self.loaded = False

        # Loading simulation data if needed
        if self.config.load:
            self.load()
            print("LOADED SUCCESSFULLY!")
            print(f"State trajectory length: {self.state_len}")

    def run(self):
        self.images_path = f"{self.root}/experiments/images/{self.name}"
        os.makedirs(self.images_path, exist_ok=True)

        if self.config.load:
            self.animation = self.init_animation(
                func=self.plot, frames=self.state_len - 1
            )
        else:
            self.animation = self.init_animation(func=self.update)
        if self.config.logging:
            self.logfile_path = f"{self.root}/experiments/logs/{self.name}.log"
            os.makedirs(self.logfile_path, exist_ok=True)
            self.logfile = open(self.logfile_path, "w")
            sys.stdout = self.logfile
        fig_manager = plt.get_current_fig_manager()
        fig_manager.window.showMaximized()
        plt.show()

    def save_animation(self):
        plt.gcf().clear()
        figure = plt.figure(figsize=(20, 10))
        animation_path = f"{self.root}/experiments/videos"
        os.makedirs(animation_path, exist_ok=True)
        animation: FuncAnimation = self.init_animation(
            func=self.plot, fig=figure, frames=self.state_len - 1
        )
        animation.save(
            f"{animation_path}/{self.name}.gif",
            fps=20,
            dpi=200,
            bitrate=1800,
            writer="pillow",
        )
        print("Animation saved!")

    @property
    @abstractmethod
    def name(self):
        pass

    @property
    @abstractmethod
    def state_len(self):
        pass

    @abstractmethod
    def summarize(self):
        pass

    @abstractmethod
    def init_containers(self):
        pass

    @abstractmethod
    def init_animation(self, func: object, fig: Figure = plt.gcf(), frames: int = None):
        pass

    @abstractmethod
    def update(self, n):
        pass

    @abstractmethod
    def plot(self, n):
        pass

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def load(self):
        pass

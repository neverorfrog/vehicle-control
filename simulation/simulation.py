from abc import ABC, abstractmethod
import os
import sys
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


class Simulation(ABC):
    
    def __init__(self, name: str, load: bool = False):
        self.name = name
        self.init_containers()
        
        # Init logfile
        self.src_dir = os.path.dirname(os.path.abspath(__file__))
        logfile_path = f'{self.src_dir}/logs/{self.name}.log'
        self.logfile = open(logfile_path, "w") 
        sys.stdout = self.logfile
        
        # Init simulation
        if load:
            self.load()
            self.animation = self.init_animation(func=self.plot,frames=self.state_len-1)
        else:
            self.animation = self.init_animation(func=self.update)
        fig_manager = plt.get_current_fig_manager()
        fig_manager.window.showMaximized()
        plt.show()
        
    def save_animation(self):
        plt.gcf().clear()
        animation: FuncAnimation = self.init_animation(func=self.plot,frames=self.state_len-1)
        animation.save(f"{self.src_dir}/videos/{self.name}.gif",fps=20, dpi=100, bitrate=1800, writer='pillow')
        print("Animation saved!")
        
    @property
    @abstractmethod
    def state_len(self):
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
    def plot(self,n):
        pass
    
    @abstractmethod
    def save(self):
        pass
    
    @abstractmethod
    def load(self):
        pass
    
    
    
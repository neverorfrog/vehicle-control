from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from model.state import FancyVector
from environment.track import Track
from utils.utils import wrap
import casadi as ca
from casadi import sin,cos,tan
from abc import abstractmethod
from abc import ABC

class RacingCar(ABC):
    def __init__(self, track: Track, length, dt):
        """
        Abstract racing Car Model
        :param track: reference path object to follow
        :param length: length of car in m
        :param dt: sampling time of model
        """
        # Precision
        self.eps = 1e-12
        # Car Parameters
        self.length = length
        # Reference Path
        self.track = track
        # Set sampling time
        self.dt = dt
        # Initialize state
        self.state: FancyVector = self.__class__.create_state()
        # Initialize input
        self.input: FancyVector = self.__class__.create_input()
        # Initialize ode 
        self._init_ode()
        # function for evaluating curvature at given s
        s = ca.MX.sym('s')
        self.curvature = ca.Function("curvature",[s],[self.track.get_curvature(s)])
        
    @classmethod
    @abstractmethod
    def create_state(cls, *args, **kwargs) -> FancyVector:
        pass
    
    @classmethod
    @abstractmethod
    def create_input(cls, *args, **kwargs) -> FancyVector:
        pass
    
    @abstractmethod
    def _init_ode(self): pass
    
    @property
    @abstractmethod
    def temporal_transition(self): pass
    
    @property
    @abstractmethod
    def spatial_transition(self): pass
    
    def drive(self, input: FancyVector):
        """
        :param input: vector of inputs
        """
        curvature = self.track.get_curvature(self.state.s)
        print(f"CURVATURE IN DRIVE: {curvature}")
        next_state = self.temporal_transition(self.state.values, input.values, curvature).full().squeeze()
        self.state = self.__class__.create_state(*next_state)
        self.input = input
        return self.state
    
    def rel2glob(self, state):
        s = state[self.state.index('s')]  
        ey = state[self.state.index('ey')] 
        epsi = state[self.state.index('epsi')]    
        track_psi = wrap(self.track.get_orientation(s))
        x = self.track.x(s) - sin(track_psi) * ey
        y = self.track.y(s) + cos(track_psi) * ey
        psi = track_psi + epsi
        return x.full().squeeze(),y.full().squeeze(),psi.full().squeeze()
    
    def plot(self, axis: Axes, state: FancyVector):
        x,y,psi = self.rel2glob(state)
        delta = state.delta
        r = self.length / 2
        
        # Draw the bicycle as a rectangle
        width = self.length
        height = self.length
        angle = wrap(psi-np.pi/2)
        rectangle = plt.Rectangle((x-np.cos(angle)*width/2-np.cos(psi)*2*width/3, y-np.sin(angle)*height/2-np.sin(psi)*2*height/3),
                                  width,height,edgecolor='black',alpha=0.7, angle=np.rad2deg(angle), rotation_point='xy')
        axis.add_patch(rectangle)
        
        # Draw four wheels as rectangles
        wheel_width = self.length / 10
        wheel_height = self.length / 4
        wheel_angle = wrap(psi+delta-np.pi/2)
        wheel_right_front = plt.Rectangle((x+np.cos(angle)*r, y+np.sin(angle)*r),width=wheel_width,height=wheel_height,angle=np.rad2deg(wheel_angle),facecolor='black')
        axis.add_patch(wheel_right_front)
        wheel_left_front = plt.Rectangle((x-np.cos(angle)*r, y-np.sin(angle)*r),width=wheel_width,height=wheel_height,angle=np.rad2deg(wheel_angle),facecolor='black')
        axis.add_patch(wheel_left_front)
        wheel_right_back = plt.Rectangle((x+np.cos(angle)*r-np.cos(psi)*width*0.6, y+np.sin(angle)*r-np.sin(psi)*height*0.6),width=wheel_width,height=wheel_height,angle=np.rad2deg(wheel_angle),facecolor='black')
        axis.add_patch(wheel_right_back)
        wheel_left_back = plt.Rectangle((x-np.cos(angle)*r-np.cos(psi)*width*0.6, y-np.sin(angle)*r-np.sin(psi)*height*0.6),width=wheel_width,height=wheel_height,angle=np.rad2deg(wheel_angle),facecolor='black')
        axis.add_patch(wheel_left_back)
        
        return x,y
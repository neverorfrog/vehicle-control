from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from omegaconf import OmegaConf
from vehicle_control.models.robot import Robot
from vehicle_control.utils.fancy_vector import FancyVector
from vehicle_control.environment.track import Track
from vehicle_control.utils.common_utils import wrap
from casadi import sin,cos
from abc import abstractmethod

class RacingCar(Robot):
    def __init__(self, config: OmegaConf, track: Track):
        """
        Abstract racing Car Model
        :param track: reference path object to follow
        :param length: length of car in m
        :param dt: sampling time of model
        """
        # Car Parameters
        self.length = config.car.l
        # Reference Path
        self.track = track
        super().__init__(config)
    
    @property
    @abstractmethod
    def spatial_transition(self): pass
    
    def drive(self, input: FancyVector):
        """
        :param input: vector of inputs
        """
        curvature = self.track.k(self.state.s)
        next_state = self.transition(self.state.values, input.values, curvature, self.dt).full().squeeze()
        self.state = self.__class__.create_state(*next_state)
        self.input = input
        return self.state
    
    def rel2glob(self, state):
        s = state[self.state.index('s')]
        ey = state[self.state.index('ey')] 
        epsi = state[self.state.index('epsi')]   
        return self.track.rel2glob(s, ey, epsi)
    
    def plot(self, axis: Axes, state: FancyVector, color = 'blue'):
        x,y,psi = self.rel2glob(state)
        delta = state[self.state.index('delta')]
        r = 1 # TODO hardcodato
        
        # Draw the bicycle as a rectangle
        width = 2 # TODO hardcodato
        height = 2 # TODO hardcodato
        angle = wrap(psi-np.pi/2)
        rectangle = plt.Rectangle(
            xy=(x-np.cos(angle)*width/2-np.cos(psi)*2*width/3, y-np.sin(angle)*height/2-np.sin(psi)*2*height/3),
            width=width,height=height,alpha=0.7,angle=np.rad2deg(angle),rotation_point='xy',color=color)
        axis.add_patch(rectangle)
        
        # Draw four wheels as rectangles
        wheel_width = self.length / 10
        wheel_height = self.length / 4
        wheel_angle = wrap(psi+delta-np.pi/2)
        wheel_right_front = plt.Rectangle((x+np.cos(angle)*r, y+np.sin(angle)*r),width=wheel_width,height=wheel_height,angle=np.rad2deg(wheel_angle),facecolor='black')
        axis.add_patch(wheel_right_front)
        wheel_left_front = plt.Rectangle((x-np.cos(angle)*r-cos(wheel_angle)*wheel_width, y-np.sin(angle)*0.9*r-sin(wheel_angle)*wheel_width),width=wheel_width,height=wheel_height,angle=np.rad2deg(wheel_angle),facecolor='black')
        axis.add_patch(wheel_left_front)
        wheel_right_back = plt.Rectangle((x+np.cos(angle)*r-np.cos(psi)*width*0.6, y+np.sin(angle)*r-np.sin(psi)*height*0.6),width=wheel_width,height=wheel_height,angle=np.rad2deg(wheel_angle),facecolor='black')
        axis.add_patch(wheel_right_back)
        wheel_left_back = plt.Rectangle((x-np.cos(angle)*r-np.cos(psi)*width*0.6-cos(wheel_angle)*wheel_width, y-np.sin(angle)*r*0.9-np.sin(psi)*height*0.6-sin(wheel_angle)*wheel_width),width=wheel_width,height=wheel_height,angle=np.rad2deg(wheel_angle),facecolor='black')
        axis.add_patch(wheel_left_back)
        
        return x,y
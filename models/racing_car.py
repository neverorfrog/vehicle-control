from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from models.robot import Robot
from utils.fancy_vector import FancyVector
from environment.track import Track
from utils.common_utils import wrap
from casadi import sin,cos
from abc import abstractmethod

class RacingCar(Robot):
    def __init__(self, config: dict, track: Track):
        """
        Abstract racing Car Model
        :param track: reference path object to follow
        :param length: length of car in m
        :param dt: sampling time of model
        """
        # Car Parameters
        self.length = config['car']['l']
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
        curvature = self.track.get_curvature(self.state.s)
        next_state = self.transition(self.state.values, input.values, curvature).full().squeeze()
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
        psi = wrap(track_psi + epsi)
        return x.full().squeeze(),y.full().squeeze(),psi.full().squeeze()
    
    
    def integrate(self,state,action,curvature,ode,h):
        '''
        RK4 integrator
        h: integration interval
        '''
        #RK4
        state_dot_1 = ode(state, action, curvature)
        state_1 = state + (h/2)*state_dot_1
        
        state_dot_2 = ode(state_1, action, curvature)
        state_2 = state + (h/2)*state_dot_2
        
        state_dot_3 = ode(state_2, action, curvature)
        state_3 = state + h*state_dot_3
        
        state_dot_4 = ode(state_3, action, curvature)
        state = state + (1/6) * (state_dot_1 + 2 * state_dot_2 + 2 * state_dot_3 + state_dot_4) * h
        
        return state
    
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
        wheel_left_front = plt.Rectangle((x-np.cos(angle)*r-cos(wheel_angle)*wheel_width, y-np.sin(angle)*r-sin(wheel_angle)*wheel_width),width=wheel_width,height=wheel_height,angle=np.rad2deg(wheel_angle),facecolor='black')
        axis.add_patch(wheel_left_front)
        wheel_right_back = plt.Rectangle((x+np.cos(angle)*r-np.cos(psi)*width*0.6, y+np.sin(angle)*r-np.sin(psi)*height*0.6),width=wheel_width,height=wheel_height,angle=np.rad2deg(wheel_angle),facecolor='black')
        axis.add_patch(wheel_right_back)
        wheel_left_back = plt.Rectangle((x-np.cos(angle)*r-np.cos(psi)*width*0.6-cos(wheel_angle)*wheel_width, y-np.sin(angle)*r-np.sin(psi)*height*0.6-sin(wheel_angle)*wheel_width),width=wheel_width,height=wheel_height,angle=np.rad2deg(wheel_angle),facecolor='black')
        axis.add_patch(wheel_left_back)
        
        return x,y
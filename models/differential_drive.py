from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from models.robot import Robot
from utils.common_utils import wrap
from utils.fancy_vector import FancyVector
import casadi as ca
from casadi import cos,sin

class DifferentialDrive(Robot):
    
    @classmethod
    def create_state(cls, *args, **kwargs):
        return DifferentialDriveState(*args, **kwargs)
    
    @classmethod
    def create_action(cls, *args, **kwargs):
        return DifferentialDriveAction(*args, **kwargs)
    
    def _init_model(self):
        
        #state variables
        x,y,psi,t = self.state.variables
        
        #input variables
        v,w = self.input.variables
        
        x_dot = v * ca.cos(psi)
        y_dot = v * ca.sin(psi)
        psi_dot = w
        t_dot = 1
        state_dot = ca.vertcat(x_dot, y_dot, psi_dot, t_dot)
        ode = ca.Function('ode', [self.state.syms,self.input.syms], [state_dot])
        integrator = self.integrate(self.state.syms,self.input.syms,ode,self.dt)
        self._transition = ca.Function('transition', [self.state.syms,self.input.syms], [integrator])
        
    def drive(self, input: FancyVector):
        """
        :param input: vector of inputs
        """
        next_state = self.transition(self.state.values, input.values).full().squeeze()
        self.state = self.__class__.create_state(*next_state)
        self.input = input
        return self.state
        
    @property
    def transition(self):
        return self._transition
    
    def plot(self, axis: Axes, state):
        x, y, psi, t = state
        r = 0.2
        
        # Plot circular shape
        circle = plt.Circle(xy=(x,y), radius=r, facecolor='orange', alpha=0.5, lw=2)
        axis.add_patch(circle)
        
        # Draw two wheels as rectangles
        wheel_angle = wrap(psi-np.pi/2)
        width = 0.05
        height = 0.15
        x_wheel_right = x+cos(wheel_angle)*r-cos(psi)*r/3-cos(wheel_angle)*width
        y_wheel_right = y+sin(wheel_angle)*r-sin(psi)*r/3-sin(wheel_angle)*width
        wheel_right = plt.Rectangle((x_wheel_right,y_wheel_right),width=width,height=height,angle=np.rad2deg(wheel_angle),facecolor='black')
        axis.add_patch(wheel_right)
        x_wheel_left = x-cos(psi)*r/3-cos(wheel_angle)*r
        y_wheel_left = y-sin(psi)*r/3-sin(wheel_angle)*r
        wheel_left = plt.Rectangle((x_wheel_left,y_wheel_left),width=width,height=height,angle=np.rad2deg(wheel_angle),facecolor='black')
        axis.add_patch(wheel_left)
    

class DifferentialDriveAction(FancyVector):
    def __init__(self, v = 0.0, w = 0.0):
        """
        :param a: longitudinal acceleration | [m/s^2]
        :param w: steering angle rate | [rad/s]
        """
        self._values = np.array([v,w])
        self._keys = ['v','w']
        self._syms = ca.vertcat(*[ca.SX.sym(self._keys[i]) for i in range(len(self._keys))])
        
    @property
    def v(self): return self.values[0] 
      
    @property
    def w(self): return self.values[1]
    
    @v.setter
    def v(self,value: float): 
        assert isinstance(value, float)
        self.values[0] = value
    
    @w.setter
    def w(self,value: float): 
        assert isinstance(value, float)
        self.values[1] = value
    
    
class DifferentialDriveState(FancyVector):
    def __init__(self, x = 0.0, y = 0.0, psi = 0.0, t = 0.0):
        """
        :param x: x coordinate | [m]
        :param y: y coordinate | [m]
        :param psi: yaw angle | [rad]
        """
        self._values = np.array([x,y,psi,t])
        self._keys = ['x','y','psi','t']
        self._syms = ca.vertcat(*[ca.SX.sym(self._keys[i]) for i in range(len(self._keys))])
    
    @property
    def x(self): return self.values[0] 
      
    @property
    def y(self): return self.values[1]
      
    @property
    def psi(self): return self.values[2]
    
    @property
    def t(self): return self.values[3]
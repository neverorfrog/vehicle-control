from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from model.state import KinematicCarInput, KinematicCarState
from environment.track import Track, Waypoint
from utils.utils import integrate, wrap
import casadi as ca
from casadi import sin,cos,tan

class KinematicCar():
    def __init__(self, track: Track, length, dt):
        """
        Kinematic Bicycle Model
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
        self.wp_id = 0
        self.current_waypoint: Waypoint = self.track.waypoints[self.wp_id]
        self.state: KinematicCarState = KinematicCarState()
        
        # Initialize input (fictituous)
        self.input: KinematicCarInput = KinematicCarInput()
        
        # Initialize dynamic model
        self._init_ode()
        
        # function for evaluating curvature at given s
        s = ca.MX.sym('s')
        self.curvature = ca.Function("curvature",[s],[self.track.get_curvature(s)])
        
    def drive(self, input: KinematicCarInput):
        """
        :param input: input vector containing [a, w]
        """
        curvature = self.track.get_curvature(self.state.s)
        next_state = self.temporal_transition(self.state.values, input.values, curvature).full().squeeze()
        self.state = KinematicCarState(*next_state)
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
        
    def _init_ode(self):
        '''Differential equations describing the temporal model'''
        
        # Input variables
        a,w = self.input.variables

        # State and auxiliary variables
        v,delta,s,ey,epsi,t = self.state.variables
        curvature = ca.SX.sym('curvature')
        ds = ca.SX.sym('ds')
        
        # TEMPORAL ODE
        v_dot = a
        delta_dot = w
        s_dot = (v * cos(epsi)) / (1 - ey * curvature)
        ey_dot = v * sin(epsi) 
        epsi_dot = v * (tan(delta)/self.length) - s_dot * curvature
        t_dot = 1
        state_dot = ca.vertcat(v_dot, delta_dot, s_dot, ey_dot, epsi_dot, t_dot)
        t_ode = ca.Function('ode', [self.state.syms,self.input.syms,curvature], [state_dot])
        t_integrator = integrate(self.state.syms,self.input.syms,curvature,t_ode,self.dt)
        self.temporal_transition = ca.Function('transition', [self.state.syms,self.input.syms,curvature], [t_integrator])
        
        # SPATIAL ODE
        v_prime = (1 - ey * curvature) / (v * np.cos(epsi)) * a
        delta_prime = (1 - ey * curvature) / (v * np.cos(epsi)) * w
        s_prime = 1
        ey_prime = (1 - ey * curvature) * ca.tan(epsi)
        epsi_prime = ((tan(delta)) / self.length) * ((1 - ey * curvature)/(cos(epsi))) - curvature
        t_prime = (1 - ey * curvature) / (v * np.cos(epsi))
        state_prime = ca.vertcat(v_prime, delta_prime, s_prime, ey_prime, epsi_prime, t_prime)
        s_ode = ca.Function('ode', [self.state.syms, self.input.syms, curvature], [state_prime])
        s_integrator = integrate(self.state.syms, self.input.syms, curvature, s_ode, h=ds)
        self.spatial_transition = ca.Function('transition', [self.state.syms,self.input.syms,curvature,ds], [s_integrator])
    
    def plot(self, axis: Axes, state: KinematicCarState):
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
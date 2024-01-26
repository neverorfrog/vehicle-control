# inspired by https://github.com/matssteinweg/Multi-Purpose-MPC

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from model.state import KinematicCarState
from environment.track import Track, Waypoint
from model.utils import wrap
import casadi as ca
from model.utils import integrate
from casadi import sin,cos,tan

class KinematicCar():
    def __init__(self, track: Track, length, dt):
        """
        Kinematic Bicycle Model
        :param track: reference path object to follow
        :param length: length of car in m
        :param width: width of car in m
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
        self.update_track_error()
        
        # Initialize dynamic model
        self._init_ode()
        
        
    def _init_ode(self):
        '''Differential equations describing the temporal model'''
        
        # Input variables
        a = ca.SX.sym('a') # driving acceleration
        w = ca.SX.sym('w') # steering angle rate
        input = ca.vertcat(a,w)

        # State and auxiliary variables
        x,y,v,psi,delta,s,ey,epsi,t = self.state.variables
        kappa = ca.SX.sym('kappa')
        
        # ODE
        x_dot = v * cos(psi)
        y_dot = v * sin(psi)
        v_dot = a
        psi_dot = v / self.length * tan(delta)
        delta_dot = w
        s_dot = (v * cos(epsi)) / (1 - ey * kappa)
        ey_dot = v * sin(epsi) 
        epsi_dot = psi_dot - s_dot * kappa
        t_dot = 1
        state_dot = ca.vertcat(x_dot, y_dot, v_dot, psi_dot, delta_dot, s_dot, ey_dot, epsi_dot, t_dot)
        ode = ca.Function('ode', [self.state.syms,input], [state_dot],{'allow_free':True})
        
        # wrapping up
        integrator = integrate(self.state.syms,input,ode,self.dt)
        self.transition = ca.Function('transition', [self.state.syms,input,kappa], [integrator])
        
    def drive(self, u):
        """
        :param u: input vector containing [a, w]
        """
        
        kappa = self.current_waypoint.kappa
        
        next_state = self.transition(self.state.values, u, kappa).full().squeeze()
        self.state = KinematicCarState(*next_state)
        self.update_waypoint()
        # print(f"s: {self.state[4]}, total length: {self.track.length}")
        # print(f"psidot: {u[0] / self.length * tan(self.state[3])}")
        # print(f"ey: {self.state[5]}")
        # print(f"epsi: {self.state[6]}")
        self.update_track_error()
        
        return self.state
    
    def update_waypoint(self) -> Waypoint:
        """
        Get closest waypoint on reference path based on car's current location.
        """
        s = self.state.s
        # Compute cumulative path length
        length_cum = np.cumsum(self.track.segment_lengths)
        # Get first index with distance larger than distance traveled by car so far
        greater_than_threshold = length_cum > s
        next_wp_id = (greater_than_threshold.searchsorted(True)) % len(length_cum)
        # Get previous index
        prev_wp_id = (next_wp_id - 1) % len(length_cum)

        # Get distance traveled for both enclosing waypoints
        s_next = length_cum[next_wp_id]
        s_prev = length_cum[prev_wp_id]
        
        if np.abs(s - s_next) < np.abs(s - s_prev):
            self.wp_id = next_wp_id
            self.current_waypoint = self.track.waypoints[next_wp_id]
        else:
            self.wp_id = prev_wp_id
            self.current_waypoint = self.track.waypoints[prev_wp_id]
            
    def update_track_error(self):
        """
        Based on current waypoint (gotten with s) and actual current x,y position,
        :return Spatial State representing the error wrt the current reference waypoint
        """
        waypoint = self.current_waypoint
        x,y = self.state.values[:2]
        psi = self.state.psi
        ey = np.cos(waypoint.psi) * (y - waypoint.y) - np.sin(waypoint.psi) * (x - waypoint.x)
        epsi = wrap(psi - waypoint.psi)
        self.state.ey = ey 
        self.state.epsi = epsi 

    
    def plot(self, axis: Axes, state: KinematicCarState):
        x = state.x
        y = state.y
        psi = state.psi
        delta = state.delta
        r = self.length / 2
        
        # Draw the bicycle as a rectangle
        width = self.length
        height = self.length
        angle = wrap(psi-np.pi/2)
        rectangle = plt.Rectangle((x-np.cos(angle)*width/2-np.cos(psi)*2*width/3, y-np.sin(angle)*height/2-np.sin(psi)*2*height/3),
                                  width,height,edgecolor='black',alpha=0.7, angle=np.rad2deg(angle), rotation_point='xy')
        axis.add_patch(rectangle)
        
        # Plot directional tick
        line_length = 1.5 * r
        line_end_x = x + line_length * np.cos(psi)
        line_end_y = y + line_length * np.sin(psi)
        axis.plot([x, line_end_x], [y, line_end_y], color='r', lw=3)
        
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
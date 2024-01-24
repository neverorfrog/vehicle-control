# inspired by https://github.com/matssteinweg/Multi-Purpose-MPC

import math
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from modeling.state import SpatialState, TemporalState
from modeling.track import Track, Waypoint
from modeling.util import wrap
import casadi as ca

class Bicycle():
    def __init__(self, track: Track, length, dt):
        """
        Bicycle Model.
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
        # Set initial distance traveled
        self.s = 0.0
        # Set sampling time
        self.dt = dt
        # Initialize spatial state
        self.spatial_state: SpatialState = SpatialState(ey=0.0,epsi=0.0)
        # Number of spatial state variables
        self.n_states = len(self.spatial_state)
        # Initialize temporal state
        self.wp_id = 0
        self.current_waypoint: Waypoint = self.track.waypoints[self.wp_id]
        self.temporal_state: TemporalState = self.s2t(reference_state=self.spatial_state,
                                       reference_waypoint=self.current_waypoint)
        
        # Initialize dynamic model
        self._init_ode()
        
    def _init_ode(self):
        # ----------- Input Variables ------------------------------------------
        v = ca.SX.sym('v') # driving acceleration
        delta = ca.SX.sym('delta') # steering angle rate
        kappa = ca.SX.sym('kappa') # the curvature is treated as an input
        self.u = ca.vertcat(v,delta,kappa)

        # --- Differential equations describing the temporal model ---------------
        tq = self.temporal_state.state_sym
        psi = self.temporal_state['psi']
        ey = self.spatial_state[0]
        epsi = self.spatial_state[1]
        x_dot = v * ca.cos(psi)
        y_dot = v * ca.sin(psi)
        psi_dot = v / self.length * ca.tan(delta)
        s_dot = (v * np.cos(epsi)) / (1 - ey * kappa)
        tqd = ca.vertcat(x_dot, y_dot, psi_dot, s_dot)
        tode = ca.Function('ode', [tq, self.u], [tqd])
        self.t_transition = ca.Function('ttransition', [tq, self.u], [self.integrate(tq,tode,self.dt)])
        
        # --- Differential equations describing the temporal model -----------------
        # sq = self.spatial_state.state_sym
        ey = self.spatial_state['ey'] # need them in variable format
        epsi = self.spatial_state['epsi']
        # t = self.spatial_state['t']
        sq = ca.vertcat(ey,epsi)
        s_dot = (v * np.cos(epsi)) / (1 - ey * kappa)
        ey_prime = (1 - ey * kappa) * ca.tan(epsi)
        epsi_prime = (self.length * ca.tan(delta)) * ((1 - ey * kappa) / (np.cos(epsi))) - kappa
        # t_prime = (1 - ey * kappa) / (v * np.cos(epsi))
        sqd = ca.vertcat(ey_prime, epsi_prime)
        sode = ca.Function('ode', [sq, self.u], [sqd])
        self.s_transition = ca.Function('stransition', [sq, self.u], [self.integrate(sq,sode,self.dt)])
        
    def integrate(self,q,ode,h):
        '''
        RK4 integrator
        h: integration interval
        '''
        qd_1 = ode(q, self.u)
        qd_2 = ode(q + (h/2)*qd_1, self.u)
        qd_3 = ode(q + (h/2)*qd_2, self.u)
        qd_4 = ode(q + h*qd_3, self.u)
        # evaluator = ca.Function('evaluator', [q, self.u], [qd_1])
        # print(evaluator)
        # exit(evaluator(self.spatial_state.state, np.array([0,0,0])))
        newq = q + (1/6) * (qd_1 + 2 * qd_2 + 2 * qd_3 + qd_4) * h
        return newq
        
        
    def drive(self, u):
        """
        Drive.
        :param u: input vector containing [v, delta]
        """

        # Get input signals
        # v, delta = u

        # # Compute temporal state derivatives
        # x_dot = v * np.cos(self.temporal_state['psi'])
        # y_dot = v * np.sin(self.temporal_state['psi'])
        # psi_dot = v / self.length * np.tan(delta)
        # temporal_derivatives = np.array([x_dot, y_dot, psi_dot])

        # Update state
        next_tstate = self.t_transition(self.temporal_state.state, u).full().squeeze()
        self.temporal_state = TemporalState(*next_tstate)
        
        next_sstate = ca.DM(self.s_transition(self.spatial_state.state, u)).full().squeeze()
        self.spatial_state = SpatialState(*next_sstate)
        
        # Compute velocity along path
        # s_dot = 1 / (1 - self.spatial_state['ey'] * self.current_waypoint.kappa) * v * np.cos(self.spatial_state['epsi'])
        # Update distance travelled along reference path
        # self.s += s_dot * self.dt
        return self.temporal_state
        
    def s2t(self, reference_waypoint: Waypoint, reference_state):
        """
        Convert spatial state to temporal state given a reference waypoint.
        :param reference_waypoint: waypoint object to use as reference
        :param reference_state: state vector as np.array to use as reference
        :return Temporal State equivalent to reference state
        """
        # Compute temporal state variables
        assert isinstance(reference_state, (SpatialState, np.ndarray)), print('Reference State type not supported!')
        x = reference_waypoint.x - reference_state[0] * np.sin(reference_waypoint.psi)
        y = reference_waypoint.y + reference_state[0] * np.cos(reference_waypoint.psi)
        psi = reference_waypoint.psi + reference_state[1]
        return TemporalState(x, y, psi)
    
    def t2s(self, reference_waypoint: Waypoint, reference_state):
        """
        Convert spatial state to temporal state. Either convert self.spatial_
        state with current waypoint as reference or provide reference waypoint
        and reference_state.
        :return Spatial State equivalent to reference state
        """
        assert isinstance(reference_state, (TemporalState, np.ndarray)), print('Reference State type not supported!')
        # Compute spatial state variables
        ey = np.cos(reference_waypoint.psi) * (reference_state[1] - reference_waypoint.y) - \
                np.sin(reference_waypoint.psi) * (reference_state[0] - reference_waypoint.x)
        epsi = wrap(reference_state[2] - reference_waypoint.psi)
        # time state can be set to zero since it's only relevant for the MPC prediction horizon
        return SpatialState(ey, epsi, 0)
    
    def get_current_waypoint(self) -> Waypoint:
        """
        Get closest waypoint on reference path based on car's current location.
        """
        s = self.temporal_state[3]
        # Compute cumulative path length
        length_cum = np.cumsum(self.track.segment_lengths)
        # Get first index with distance larger than distance traveled by car so far
        greater_than_threshold = length_cum > s
        next_wp_id = greater_than_threshold.searchsorted(True)
        # Get previous index
        prev_wp_id = next_wp_id - 1

        # Get distance traveled for both enclosing waypoints
        s_next = length_cum[next_wp_id]
        s_prev = length_cum[prev_wp_id]
        
        if np.abs(s - s_next) < np.abs(s - s_prev):
            self.wp_id = next_wp_id
            self.current_waypoint = self.track.waypoints[next_wp_id]
        else:
            self.wp_id = prev_wp_id
            self.current_waypoint = self.track.waypoints[prev_wp_id]
        return self.current_waypoint
    
    def plot(self, axis: Axes, q = None):
        x,y,psi,s = self.temporal_state.state if q is None else q
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
        wheel_angle = wrap(psi-np.pi/2)
        wheel_right_front = plt.Rectangle((x+np.cos(angle)*r, y+np.sin(angle)*r),width=wheel_width,height=wheel_height,angle=np.rad2deg(wheel_angle),facecolor='black')
        axis.add_patch(wheel_right_front)
        wheel_left_front = plt.Rectangle((x-np.cos(angle)*r, y-np.sin(angle)*r),width=wheel_width,height=wheel_height,angle=np.rad2deg(wheel_angle),facecolor='black')
        axis.add_patch(wheel_left_front)
        wheel_right_back = plt.Rectangle((x+np.cos(angle)*r-np.cos(psi)*width*0.6, y+np.sin(angle)*r-np.sin(psi)*height*0.6),width=wheel_width,height=wheel_height,angle=np.rad2deg(wheel_angle),facecolor='black')
        axis.add_patch(wheel_right_back)
        wheel_left_back = plt.Rectangle((x-np.cos(angle)*r-np.cos(psi)*width*0.6, y-np.sin(angle)*r-np.sin(psi)*height*0.6),width=wheel_width,height=wheel_height,angle=np.rad2deg(wheel_angle),facecolor='black')
        axis.add_patch(wheel_left_back)
        
        return x,y
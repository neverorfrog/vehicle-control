import casadi as ca
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from modeling.track import Track
from modeling.util import *
from casadi import cos, sin, tan, atan2

class Unicycle():
    '''
    state = [s,ey,epsi,v,psi,t]
    x : absolute x position
    y : absolute y position
    s : distance traveled along the road
    ey : spatial error to path
    epsi : angular error to path
    v : translational velocity
    psi : yaw angle (also called theta usually)
    t : time (we need it for MPC planning phase)
    '''
    def __init__(self, track: Track, l: float = 0.5):
        self.l = l
        self.track = track # the track itself becomes part of the model
        
        # ----------- State Variables --------------------------------------
        self.x      = ca.MX.sym('x')
        self.y      = ca.MX.sym('y')
        self.psi    = ca.MX.sym('psi')
        self.s      = ca.MX.sym('s')
        self.ey     = ca.MX.sym('ey')
        self.epsi   = ca.MX.sym('epsi')
        self.t      = ca.MX.sym('t')
        self.q      = ca.vertcat(self.x, self.y, self.psi, self.s, self.ey, self.epsi, self.t)
        self.q_keys = ['x','y','psi','s','ey','epsi','t']
        self.q_len  = len(self.q_keys)
        
        # ----------- Input Variables ------------------------------------------
        self.v = ca.MX.sym('v') # driving acceleration
        self.w = ca.MX.sym('w') # steering angle rate
        self.u = ca.vertcat(self.v,self.w)
        self.u_len = 2

        # --- Differential equations describing the model ----------------------
        self.k = ca.MX.sym('k') # curvature (where do we compute it?)
        self.x_dot      = self.v * cos(self.psi)
        self.y_dot      = self.v * sin(self.psi)
        self.psi_dot    = self.w
        self.s_dot      = (self.v * cos(self.epsi)) / (1 - self.k*self.ey)
        self.ey_dot     = self.v * sin(self.epsi)
        self.epsi_dot   = self.psi_dot - self.s_dot * self.k
        self.t_dot      = 1
        self.qd = ca.vertcat(self.x_dot, self.y_dot, self.psi_dot, self.s_dot, self.ey_dot, self.epsi_dot, self.t_dot)
        self.ode = ca.Function('ode', [self.q, self.u, self.k], [self.qd])
    
    def integrate(self,h):
        '''
        RK4 integrator
        h: integration interval
        '''
        q = self.q
        qd_1 = self.ode(q, self.u, self.k)
        qd_2 = self.ode(q + (h/2)*qd_1, self.u, self.k)
        qd_3 = self.ode(q + (h/2)*qd_2, self.u, self.k)
        qd_4 = self.ode(q + h*qd_3, self.u, self.k)
        q += (1/6) * (qd_1 + 2 * qd_2 + 2 * qd_3 + qd_4) * h
        
        return q

    def extract_pose(self, q = None):
        if q is None: return np.array([0,0,0])
        return np.array([q['x'], q['y'], q['psi']])
        
    def plot(self, axis: Axes, q):
        x,y,psi = q
        r = 0.2
        
        # Plot circular shape
        circle = plt.Circle(xy=(x,y), radius=r, edgecolor='b', facecolor='none', lw=2)
        axis.add_patch(circle)
        
        # Plot directional tick
        line_length = 1.5 * r
        line_end_x = x + line_length * np.cos(psi)
        line_end_y = y + line_length * np.sin(psi)
        axis.plot([x, line_end_x], [y, line_end_y], color='r', lw=3)
        
        # Draw two wheels as rectangles
        wheel_angle = wrap(psi-np.pi/2)
        wheel_right = plt.Rectangle((x+np.cos(wheel_angle)*r, y+np.sin(wheel_angle)*r),width=0.05,height=0.15,angle=np.rad2deg(wheel_angle),facecolor='black')
        axis.add_patch(wheel_right)
        wheel_left = plt.Rectangle((x-np.cos(wheel_angle)*r, y-np.sin(wheel_angle)*r),width=0.05,height=0.15,angle=np.rad2deg(wheel_angle),facecolor='black')
        axis.add_patch(wheel_left)
        return x,y     
import casadi as ca
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from modeling.track import Track
from modeling.util import *
from casadi import cos, sin, tan, atan2

class Car():
    '''
    state = [v,r,s,ey,epsi,delta]
    v : translational velocity
    psi : yaw angle
    t : time (we need it for MPC planning phase)
    ey : spatial error to path
    epsi : angular error to path
    delta : steering angle
    s : distance traveled along the road
    '''
    def __init__(self, track: Track, l: float = 0.5):
        self.l = l
        self.track = track # the track itself becomes part of the model
        
        # ----------- State Variables --------------------------------------
        self.v = ca.MX.sym('v')
        self.psi = ca.MX.sym('psi')
        self.t = ca.MX.sym('t')
        self.ey = ca.MX.sym('ey')
        self.epsi = ca.MX.sym('epsi')
        self.delta = ca.MX.sym('delta')
        self.s = ca.MX.sym('s')
        self.q = ca.vertcat(self.v,self.psi,self.t,self.ey,self.epsi,self.delta,self.s)
        self.q_keys = ['v','psi','t','ey','epsi','delta','s']
        self.q_len = len(self.q_keys)
        
        # ----------- Input Variables ------------------------------------------
        self.a = ca.MX.sym('a') # driving acceleration
        self.w = ca.MX.sym('w') # steering angle rate
        self.u = ca.vertcat(self.a,self.w)
        self.u_len = 2

        # --- Differential equations describing the model ----------------------
        self.k = ca.MX.sym('k') # curvature (where do we compute it?)
        v_dot       = self.a
        psi_dot     = self.v * tan(self.delta) / self.l
        t_dot       = 1
        ey_dot      = self.v * sin(self.epsi)
        s_dot       = (self.v * cos(self.epsi)) / (1 - self.ey*self.k)
        epsi_dot    = psi_dot - s_dot * self.k
        delta_dot   = self.w
        self.qd     = ca.vertcat(v_dot,psi_dot,t_dot,ey_dot,epsi_dot,delta_dot,s_dot)
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
        
    def plot(self, axis: Axes, q):
        x,y,psi,delta = q
        r = 0.2
        
        # Draw the bicycle as a rectangle
        width = 0.5
        height = 0.5
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
        wheel_angle = wrap(psi+delta-np.pi/2)
        wheel_right_front = plt.Rectangle((x+np.cos(angle)*r, y+np.sin(angle)*r),width=0.05,height=0.15,angle=np.rad2deg(wheel_angle),facecolor='black')
        axis.add_patch(wheel_right_front)
        wheel_left_front = plt.Rectangle((x-np.cos(angle)*r, y-np.sin(angle)*r),width=0.05,height=0.15,angle=np.rad2deg(wheel_angle),facecolor='black')
        axis.add_patch(wheel_left_front)
        wheel_right_back = plt.Rectangle((x+np.cos(angle)*r-np.cos(psi)*2*width/3, y+np.sin(angle)*r-np.sin(psi)*2*height/3),width=0.05,height=0.15,angle=np.rad2deg(wheel_angle),facecolor='black')
        axis.add_patch(wheel_right_back)
        wheel_left_back = plt.Rectangle((x-np.cos(angle)*r-np.cos(psi)*2*width/3, y-np.sin(angle)*r-np.sin(psi)*2*height/3),width=0.05,height=0.15,angle=np.rad2deg(wheel_angle),facecolor='black')
        axis.add_patch(wheel_left_back)
        return x,y      
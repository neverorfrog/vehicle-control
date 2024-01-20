import casadi as ca
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from modeling.track import Track
from modeling.util import *
from modeling.robot import *
from casadi import cos, sin, tan

class Bicycle(Robot):
    def __init__(self, track: Track, l: float = 0.5):
        self.l = l
        self.track = track
        '''the track itself becomes part of the bicycle model'''
        
        # ----------- State Variables ----------------------
        x = ca.SX.sym('x')
        '''x position'''
        
        y = ca.SX.sym('y')
        '''y position'''
        
        theta = ca.SX.sym('theta') 
        '''longitudinal direction'''
        
        delta = ca.SX.sym('phi') 
        '''steering angle'''
        
        s = ca.SX.sym('s') 
        '''distance along the road descriptor'''
        
        e = ca.SX.sym('e') 
        '''lateral distance to the path'''
        
        phi = ca.SX.sym('phi') 
        '''orientation error wrt the path (course error)'''
        
        k = ca.SX.sym('k')
        '''curvature of the path'''
        
        q = ca.vertcat(x,y,theta,delta,s,e,phi,k)
        self.state_labels=['x','y','theta','delta']
        
        # ----------- Input Variables -----------------------
        v = ca.SX.sym('v') 
        '''longitdinal velocity'''
        
        w = ca.SX.sym('w')
        '''steering velocity'''
        
        u = ca.vertcat(v,w)
        self.input_labels=['v','w']

        # --- Differential equations describing the model ----
        x_dot = v * cos(theta)
        y_dot = v * sin(theta)
        delta_dot = w
        theta_dot = v * (tan(delta) / l)
        s_dot = (v * cos(phi)) / (1 - e*k)
        e_dot = v * sin(phi)
        phi_dot = w - k*s_dot
        k_dot = 1
        
        qd = ca.vertcat(x_dot, y_dot, theta_dot, delta_dot, s_dot, e_dot, phi_dot, k_dot)
        super().__init__(q,u,qd)
        
    def plot(self, axis: Axes, q):
        x,y,theta,delta,s,e,phi,k = q
        r = 0.2
        
        # Draw the bicycle as a rectangle
        width = 0.5
        height = 0.5
        angle = wrap(theta-np.pi/2)
        rectangle = plt.Rectangle((x-np.cos(angle)*width/2-np.cos(theta)*2*width/3, y-np.sin(angle)*height/2-np.sin(theta)*2*height/3),
                                  width,height,edgecolor='black',alpha=0.7, angle=np.rad2deg(angle), rotation_point='xy')
        axis.add_patch(rectangle)
        
        # Plot directional tick
        line_length = 1.5 * r
        line_end_x = x + line_length * np.cos(theta)
        line_end_y = y + line_length * np.sin(theta)
        axis.plot([x, line_end_x], [y, line_end_y], color='r', lw=3)
        
        # Draw four wheels as rectangles
        wheel_angle = wrap(theta+delta-np.pi/2)
        wheel_right_front = plt.Rectangle((x+np.cos(angle)*r, y+np.sin(angle)*r),width=0.05,height=0.15,angle=np.rad2deg(wheel_angle),facecolor='black')
        axis.add_patch(wheel_right_front)
        wheel_left_front = plt.Rectangle((x-np.cos(angle)*r, y-np.sin(angle)*r),width=0.05,height=0.15,angle=np.rad2deg(wheel_angle),facecolor='black')
        axis.add_patch(wheel_left_front)
        wheel_right_back = plt.Rectangle((x+np.cos(angle)*r-np.cos(theta)*2*width/3, y+np.sin(angle)*r-np.sin(theta)*2*height/3),width=0.05,height=0.15,angle=np.rad2deg(wheel_angle),facecolor='black')
        axis.add_patch(wheel_right_back)
        wheel_left_back = plt.Rectangle((x-np.cos(angle)*r-np.cos(theta)*2*width/3, y-np.sin(angle)*r-np.sin(theta)*2*height/3),width=0.05,height=0.15,angle=np.rad2deg(wheel_angle),facecolor='black')
        axis.add_patch(wheel_left_back)
        return x,y  
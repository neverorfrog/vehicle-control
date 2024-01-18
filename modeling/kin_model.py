from collections import deque
import casadi as ca
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np

class Model():
    '''
        Defines the ODE of a dynamic system
        q (state), u (input):    casadi expression that have been used to define the dynamics qd
        qd (state_dot):          casadi expr defining the rhs of the ode 
    '''
    def __init__(self, q, u, qd):
        self.q = q; self.u = u; self.qd = qd
        self.q_len = q.shape[0]
        self.u_len = u.shape[0]
        self.transition_function = ca.Function('qdot', [q, u], [qd])
        
    def plot(axis: Axes, q):
        '''Plots the actual shape of the robot'''
        pass
        
    def RK4(self,dt,integration_steps=10):
        '''
        RK4 integrator
        dt:             integration interval
        N_steps:        number of integration steps per integration interval, default:1
        '''
        h = dt/integration_steps
        current_q = self.q
        
        for _ in range(integration_steps):
            k_1 = self.transition_function(current_q, self.u)
            k_2 = self.transition_function(current_q + (dt/2)*k_1, self.u)
            k_3 = self.transition_function(current_q + (dt/2)*k_2, self.u)
            k_4 = self.transition_function(current_q + dt*k_3, self.u)
            current_q += (1/6) * (k_1 + 2 * k_2 + 2 * k_3 + k_4) * h
        return current_q
    
# Every model has to define
# q, u, qd by passing it to super() constructor
# q_labels, u_labels (for plotting purposes)
# a plot function for the actual robot visualization
        
class DifferentialDrive(Model):
    def __init__(self):
        # state
        x = ca.SX.sym('x') # q 1
        y = ca.SX.sym('y') # q 2
        theta = ca.SX.sym('theta') # q 3
        q = ca.vertcat(x,y,theta)
        self.state_labels=['x','y','theta']
        
        # input
        v = ca.SX.sym('v') # u 1
        w = ca.SX.sym('w') # u 2
        u = ca.vertcat(v,w)
        self.input_labels=['v','w']
        
        # ODE
        x_dot = v * ca.cos(theta)
        y_dot = v * ca.sin(theta)
        theta_dot = w
        qd = ca.vertcat(x_dot, y_dot, theta_dot)
        
        super().__init__(q,u,qd)
    
    def plot(self, axis: Axes, q, window: deque):
        x,y,theta = q
        r = 0.2
        window.append([x,y])
        
        # Plot circular shape
        circle = plt.Circle(xy=(x,y), radius=r, edgecolor='b', facecolor='none', lw=2)
        axis.add_patch(circle)
        
        # Plot directional tick
        line_length = 1.5 * r
        line_end_x = x + line_length * np.cos(theta)
        line_end_y = y + line_length * np.sin(theta)
        axis.plot([x, line_end_x], [y, line_end_y], color='r', lw=3)
        
        # Plot last window points
        window_np = np.array(window)
        axis.plot(window_np[:,0],window_np[:,1])
        
        
from enum import Enum
class Traction(Enum):
    RW = 1
    FW = 2

class Bicycle(Model):
    def __init__(self, l: float = 0.5, traction: Traction = Traction.RW):
        # Variables
        x = ca.SX.sym('x') # q 1
        y = ca.SX.sym('y') # q 2
        theta = ca.SX.sym('theta') # q 3
        phi = ca.SX.sym('phi') # q 4
        q = ca.vertcat(x,y,theta,phi)
        
        v = ca.SX.sym('v') # u 1
        w = ca.SX.sym('w') # u 2
        u = ca.vertcat(v,w)
        
        if traction is Traction.FW:
            print("TODO")
        elif traction is Traction.RW:
            x_dot = v * ca.cos(theta)
            y_dot = v * ca.sin(theta)
            theta_dot = v * (ca.tan(phi) / l)
            phi_dot = w
        qd = ca.vertcat(x_dot, y_dot, theta_dot, phi_dot)
        super().__init__(q,u,qd)
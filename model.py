import casadi as ca
from utils import *


class Model(Parameters):
    """
        Defines the ODE of a dynamic system
        state, input:   casadi expression that have been used to define the dynamics state_dot
        state_dot:      casadi expr defining the rhs of the ode 
    """
    def __init__(self, state, input, state_dot, dt, integration_steps):
        self.save_parameters()
        self.state_len = state.shape[0]
        self.input_len = input.shape[0]
        self.discrete_ode = ca.Function(
            'discrete_ode', 
            [state, input], 
            [RK4(state,input,state_dot,dt,integration_steps)])
        
    def step(self, state_k, input_k):
        """
            - Given current (kth) state and input
            - Applies it for dt (given in Model construction)
            - Return numpy array of next state
        """
        return self.discrete_ode(state_k,input_k).full()
        
class DifferentialDrive(Model):
    def __init__(self, params: dict):
        # Variables
        x = ca.SX.sym('x') # state 1
        y = ca.SX.sym('y') # state 2
        theta = ca.SX.sym('theta') # state 3
        state = ca.vertcat(x,y,theta)
        v = ca.SX.sym('v') # input 1
        w = ca.SX.sym('w') # input 2
        input = ca.vertcat(v,w)
        
        # ODE
        x_dot = v * ca.cos(theta)
        y_dot = v * ca.sin(theta)
        theta_dot = w
        state_dot = ca.vertcat(x_dot, y_dot, theta_dot)
        
        super().__init__(state,input,state_dot,dt=params['dt'],integration_steps=params['integration_steps'])
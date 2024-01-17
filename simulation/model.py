import casadi as ca

class Model():
    """
        Defines the ODE of a dynamic system
        q (state), u (input):    casadi expression that have been used to define the dynamics qd
        qd (state_dot):          casadi expr defining the rhs of the ode 
    """
    def __init__(self, q, u, qd):
        self.q = q; self.u = u; self.qd = qd
        self.q_len = q.shape[0]
        self.u_len = u.shape[0]
        
    def RK4(self,dt,integration_steps=10):
        '''
        RK4 integrator
        dt:             integration interval
        N_steps:        number of integration steps per integration interval, default:1
        '''
        h = dt/integration_steps
        current_state = self.q
        transition_function = ca.Function('xdot', [self.q, self.u], [self.qd])
        
        for _ in range(integration_steps):
            k_1 = transition_function(current_state, self.u)
            k_2 = transition_function(current_state + (dt/2)*k_1, self.u)
            k_3 = transition_function(current_state + (dt/2)*k_2, self.u)
            k_4 = transition_function(current_state + dt*k_3, self.u)

            current_state += (1/6) * (k_1 + 2 * k_2 + 2 * k_3 + k_4) * h

        return current_state 
        
class DifferentialDrive(Model):
    """
        Defines the ODE of a differential drive
        q (state), u (input):    casadi expression that have been used to define the dynamics qd
        qd (state_dot):      casadi expr defining the rhs of the ode 
    """
    def __init__(self):
        # Variables
        x = ca.SX.sym('x') # q 1
        y = ca.SX.sym('y') # q 2
        theta = ca.SX.sym('theta') # q 3
        q = ca.vertcat(x,y,theta)
        v = ca.SX.sym('v') # u 1
        w = ca.SX.sym('w') # u 2
        u = ca.vertcat(v,w)
        
        # ODE
        x_dot = v * ca.cos(theta)
        y_dot = v * ca.sin(theta)
        theta_dot = w
        qd = ca.vertcat(x_dot, y_dot, theta_dot)
        
        super().__init__(q,u,qd)
  
from enum import Enum
class Traction(Enum):
    RW = 1
    FW = 2

class Bicycle(Model):
    def __init__(self, traction: Traction):
        # Variables
        x = ca.SX.sym('x') # q 1
        y = ca.SX.sym('y') # q 2
        theta = ca.SX.sym('theta') # q 3
        phi = ca.SX.sym('phi') # q 4
        q = ca.vertcat(x,y,theta,phi)
        
        # TODO Implement bicicyle kinematic model
        v = ca.SX.sym('v') # u 1
        w = ca.SX.sym('w') # u 2
        u = ca.vertcat(v,w)
        
        if traction is Traction.FW:
            print("ciao")
        elif traction is Traction.RW:
            print("hola")
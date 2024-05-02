from environment.trajectory import Trajectory
import numpy as np
from controllers.controller import Controller
from casadi import sin,cos
from models.differential_drive import DifferentialDrive, DifferentialDriveAction
import casadi as ca

class FBL(Controller):
    def __init__(self, kp: np.ndarray, kd: np.ndarray, b = 0.1):
        super().__init__(kp, kd)
        self.b = b
        
    def command(self, robot: DifferentialDrive, reference: Trajectory):
        state = robot.state
        # point at distance b from center
        x_b = state.x + self.b * cos(state.psi)
        y_b = state.y + self.b * sin(state.psi)
        
        ref = reference.update(state.t)
            
        # intermediate control signal
        e_p = ref['p'] - [x_b,y_b]
        u_io = ref['pd'] + self.kp*e_p
        
        # linearization
        inverse_decoupling_matrix = np.array([
            [cos(state.psi), sin(state.psi)],
            [-sin(state.psi)/self.b, cos(state.psi)/self.b]])

        action = np.matmul(inverse_decoupling_matrix,u_io)
    
        return DifferentialDriveAction(action[0], action[1]), ref['p'], e_p 

class DFBL(Controller):
    def __init__(self, kp: np.ndarray, kd: np.ndarray):
        super().__init__(kp, kd)
        v = ca.SX.sym('v') 
        a = ca.SX.sym('a')
        v_dot = a
        self.ode = ca.Function('ode', [v,a], [v_dot])
        integrator = self.integrate(v,a,h=0.05)
        self.v_transition = ca.Function('transition', [v,a], [integrator])
        
    def command(self, robot: DifferentialDrive, reference: Trajectory):
        state = robot.state
        
        # calculating velocity
        input = robot.input
        xd = cos(state.psi) * input.v
        yd = sin(state.psi) * input.v
        
        ref = reference.update(state.t)
        e_p = ref['p'] - [state.x,state.y]
        e_d = ref['pd'] - [xd,yd]
        u_io = ref['pdd'] + e_p*self.kp + e_d*self.kd
                
        inverse_decoupling_matrix = np.array([
            [cos(state.psi), sin(state.psi)],
            [-sin(state.psi)/input.v, cos(state.psi)/input.v]])
        
        a_w = np.matmul(inverse_decoupling_matrix,u_io)
        v = self.v_transition(input.v, a_w[0]).full().squeeze()
        return DifferentialDriveAction(v, a_w[1]), ref['p'], e_p 
    
    def integrate(self,v,a,h):
        '''
        RK4 integrator
        h: integration interval
        '''
        vd_1 = self.ode(v,a)
        vd_2 = self.ode(v + (h/2)*vd_1,a)
        vd_3 = self.ode(v + (h/2)*vd_2,a)
        vd_4 = self.ode(v + h*vd_3,a)
        new_v = v + (1/6) * (vd_1 + 2 * vd_2 + 2 * vd_3 + vd_4) * h
        return new_v
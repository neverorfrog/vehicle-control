import numpy as np
from model.racing_car import RacingCar
from model.state import KinematicCarInput, KinematicCarState
from utils.utils import integrate
import casadi as ca
from casadi import sin,cos,tan

class KinematicCar(RacingCar):        
    @classmethod
    def create_state(cls, *args, **kwargs):
        return KinematicCarState(*args, **kwargs)
    
    @classmethod
    def create_input(cls, *args, **kwargs):
        return KinematicCarInput(*args, **kwargs)
        
    def _init_ode(self):
        '''Differential equations describing the model'''
        
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
        self._temporal_transition = ca.Function('transition', [self.state.syms,self.input.syms,curvature], [t_integrator])
        
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
        self._spatial_transition = ca.Function('transition', [self.state.syms,self.input.syms,curvature,ds], [s_integrator])
        
    @property
    def temporal_transition(self):
        return self._temporal_transition
    
    @property
    def spatial_transition(self):
        return self._spatial_transition
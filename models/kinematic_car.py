import numpy as np
from models.racing_car import RacingCar
from utils.fancy_vector import FancyVector
import casadi as ca
from casadi import sin,cos,tan
from utils.integrators import EulerIntegrator

class KinematicCar(RacingCar):        
    @classmethod
    def create_state(cls, *args, **kwargs):
        return KinematicCarState(*args, **kwargs)
    
    @classmethod
    def create_input(cls, *args, **kwargs):
        return KinematicCarInput(*args, **kwargs)
    
    def print(self, state, input):
        pass
        
    def _init_model(self):
        '''Differential equations describing the model'''
        
        # =========== Input variables ===================================
        a,w = self.input.variables

        # =========== State and auxiliary variables ===================================
        v,delta,ey,epsi,_,_ = self.state.variables
        curvature = ca.SX.sym('curvature')
        ds = ca.SX.sym('ds')
        dt = ca.SX.sym('dt')
        
        # =========== Temporal ODE ===================================
        v_dot = a
        delta_dot = w
        s_dot = (v * cos(epsi)) / (1 - ey*curvature)
        ey_dot = v * sin(epsi) 
        epsi_dot = v * (tan(delta)/self.length) - s_dot*curvature
        t_dot = 1
        state_dot = ca.vertcat(v_dot, delta_dot, ey_dot, epsi_dot, s_dot, t_dot)
        time_integrator = EulerIntegrator(self.state.syms,self.input.syms,curvature,state_dot,dt)
        self._temporal_transition = time_integrator.step
        
        # =========== Spatial ODE ===================================
        v_prime = ((1 - ey*curvature) / (v*cos(epsi))) * a #dv/dt * dt/ds = dv/ds
        delta_prime = ((1 - ey*curvature) / (v * cos(epsi))) * w #dw/dt * dt/ds = dw/ds
        ey_prime = (1 - ey*curvature) * tan(epsi)
        epsi_prime = (tan(delta) / self.length) * ((1 - ey*curvature)/cos(epsi)) - curvature
        s_prime = 1
        t_prime = (1 - ey*curvature) / (v*cos(epsi))
        state_prime = ca.vertcat(v_prime, delta_prime, ey_prime, epsi_prime, s_prime, t_prime)
        space_integrator = EulerIntegrator(self.state.syms,self.input.syms,curvature,state_prime,ds)
        self._spatial_transition = space_integrator.step
        
    @property
    def transition(self):
        return self._temporal_transition
    
    @property
    def spatial_transition(self):
        return self._spatial_transition
    
class KinematicCarInput(FancyVector):
    def __init__(self, a = 0.0, w = 0.0):
        """
        :param a: longitudinal acceleration | [m/s^2]
        :param w: steering angle rate | [rad/s]
        """
        self._values = np.array([a,w])
        self._keys = ['a','w']
        self._syms = ca.vertcat(*[ca.SX.sym(self._keys[i]) for i in range(len(self._keys))])
        self._labels = [r'$a$',r'$\omega$']
        
    @property
    def a(self): return self.values[0] 
      
    @property
    def w(self): return self.values[1]
    
    @a.setter
    def a(self,value: float): 
        assert isinstance(value, float)
        self.values[0] = value
    
    @w.setter
    def w(self,value: float): 
        assert isinstance(value, float)
        self.values[1] = value
  
class KinematicCarState(FancyVector):
    def __init__(self, v = 0.0, delta = 0.0, ey = 0.0, epsi = 0.0, s = 0.0, t = 0.0):
        """
        :param v: velocity in global coordinate system | [m/s]
        :param delta: steering angle | [rad]
        :param ey: orthogonal deviation from center-line | [m]
        :param epsi: yaw angle relative to path | [rad]
        :param s: curvilinear abscissa | [m]
        """
        self._values = np.array([v,delta,ey,epsi,s,t])
        self._keys = ['v','delta','ey','epsi','s','t']
        self._syms = ca.vertcat(*[ca.SX.sym(self._keys[i]) for i in range(len(self._keys))])
        
    @property
    def v(self): return self.values[0] 
      
    @property
    def delta(self): return self.values[1]
    
    @property
    def ey(self): return self.values[2]
    
    @ey.setter
    def ey(self, value): self.values[2] = value
    
    @property
    def epsi(self): return self.values[3]
    
    @epsi.setter
    def epsi(self, value): self.values[3] = value  
    
    @property
    def s(self): return self.values[4]
    
    @property
    def t(self): return self.values[5]
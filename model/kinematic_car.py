import numpy as np
from model.racing_car import RacingCar
from utils.fancy_vector import FancyVector
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
        next_curvature = ca.SX.sym('next_curvature')
        ds = ca.SX.sym('ds')
        s = ca.MX.sym('s')
        # self.state['s'] = s
        
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
        s_integrator = self.integrate(self.state.syms, self.input.syms, curvature, next_curvature, s_ode, h=ds)
        self._spatial_transition = ca.Function('transition', [self.state.syms,self.input.syms,curvature,next_curvature,ds], [s_integrator])
        
    @property
    def temporal_transition(self):
        return self._temporal_transition
    
    @property
    def spatial_transition(self):
        return self._spatial_transition
    
    def integrate(self,q,u,curvature,next_curvature,ode,h):
        '''
        RK4 integrator
        h: integration interval
        '''
        qd_1 = ode(q, u, curvature)
        # print("next_curvature")
        # hi = q + (h/2)*qd_1
        # self.track.get_curvature(hi)
        qd_2 = ode(q + (h/2)*qd_1, u, curvature)
        qd_3 = ode(q + (h/2)*qd_2, u, curvature)
        qd_4 = ode(q + h*qd_3, u, curvature)
        newq = q + (1/6) * (qd_1 + 2 * qd_2 + 2 * qd_3 + qd_4) * h
        return newq
    
class KinematicCarInput(FancyVector):
    def __init__(self, a = 0.0, w = 0.0):
        """
        :param a: longitudinal acceleration | [m/s^2]
        :param w: steering angle rate | [rad/s]
        """
        self._values = np.array([a,w])
        self._keys = ['a','w']
        self._syms = ca.vertcat(*[ca.SX.sym(self._keys[i]) for i in range(len(self._keys))])
        
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
        
    @property
    def values(self): return self._values
    
    @property
    def syms(self): return self._syms
    
    @property
    def keys(self): return self._keys
    
    @classmethod
    def create(cls, *args, **kwargs):
        return cls(*args, **kwargs)

class KinematicCarState(FancyVector):
    def __init__(self, v = 0.0, delta = 0.0, s = 0.0, ey = 0.0, epsi = 0.0, t = 0.0):
        """
        :param v: velocity in global coordinate system | [m/s]
        :param psi: yaw angle | [rad]
        :param delta: steering angle | [rad]
        :param s: curvilinear abscissa | [m]
        :param ey: orthogonal deviation from center-line | [m]
        :param epsi: yaw angle relative to path | [rad]
        :param t: time | [s]
        """
        self._values = np.array([v,delta,s,ey,epsi,t])
        self._keys = ['v','delta','s','ey','epsi','t']
        self._syms = ca.vertcat(*[ca.SX.sym(self._keys[i]) for i in range(len(self._keys))])
    
    @property
    def v(self): return self.values[0] 
      
    @property
    def delta(self): return self.values[1]
      
    @property
    def s(self): return self.values[2]
    
    @property
    def ey(self): return self.values[3]
    
    @ey.setter
    def ey(self, value): self.values[3] = value
    
    @property
    def epsi(self): return self.values[4]
    
    @epsi.setter
    def epsi(self, value): self.values[4] = value
    
    @property
    def t(self): return self.values[5]
    
    @property
    def values(self): return self._values
    
    @property
    def syms(self): return self._syms
    
    @property
    def keys(self): return self._keys
    
    @classmethod
    def create(cls, *args, **kwargs):
        return cls(*args, **kwargs)
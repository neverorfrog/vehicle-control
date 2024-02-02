import casadi as ca
from model.racing_car import RacingCar
from utils.fancy_vector import FancyVector
from utils.common_utils import *
from collections import namedtuple

class DynamicPointMass(RacingCar):
    

    def set_state_from_singletrack_state(self, state):
        s = state.s
        V = ca.sqrt(state.Ux**2 + state.Uy**2)
        ey = state.ey
        epsi = ca.arctan(state.Uy/state.Uy) + state.epsi
        self.state = DynamicPointMassState(V=V, s=s, ey=ey, epsi=epsi)


    @classmethod
    def create_state(cls, *args, **kwargs):
        return DynamicPointMassState(*args, **kwargs)
    
    @classmethod
    def create_input(cls, *args, **kwargs):
        return DynamicPointMassInput(*args, **kwargs)
    
    def _init_ode(self):
        g = 9.88
        # Input variables
        Fx, Fy = self.input.variables

        # State and auxiliary variables
        V,s,ey,epsi,t = self.state.variables
        curvature = ca.SX.sym('curvature')
        ds = ca.SX.sym('ds')

        #parameters: mass, distance of CG from front (a) and rear (b) axis, height of CG, yaw moment of inertia
        Parameters = namedtuple('Parameters', ['m'])
        p = Parameters(1000)
        #TODO theta and phi are road grade and bank angle, but for now we assume flat track
        theta = 0 
        phi = 0

        Fb = 0 #-p.m*g*ca.cos(theta)*ca.sin(phi) TODO if you want to change the angle modify this
        Fn = -p.m*g*1 #ca.cos(theta) is 1 for theta=0, might aswell not write it
        
        Crr = 0.014 #https://en.wikipedia.org/wiki/Rolling_resistance
        Frr = Crr*Fn #rolling resistance = coefficient*normal force (not specified in the paper)
        
        #All the forces introduced above are constant, as the various coefficient are constant and the ground is always flat
            #Fd depends on the velocity instead, so we define a casadi function (Is this the correct methodology?)
        Cd = 0.25 #https://en.wikipedia.org/wiki/Automobile_drag_coefficient#:~:text=The%20average%20modern%20automobile%20achieves,a%20Cd%3D0.35%E2%80%930.45.
        Fd = ca.Function("Fd_Function",[V], [Frr + Cd*V**2 - 0]) #p.m*g*ca.sin(theta) 
        
        
        # TEMPORAL ODE (equations 1a to 1f)
        V_dot = (Fx - Fd(V))/p.m
        s_dot = V*ca.cos(epsi)/(1-curvature*ey)
        ey_dot = V*ca.sin(epsi)
        epsi_dot = (Fy + Fb)/(p.m*V) - curvature*s_dot
        t_dot = 1
        state_dot = ca.vertcat(V_dot,  s_dot, ey_dot, epsi_dot, t_dot)
        t_ode = ca.Function('ode', [self.state.syms,self.input.syms,curvature], [state_dot])
        t_integrator = integrate(self.state.syms,self.input.syms,curvature,t_ode,self.dt)
        self._temporal_transition = ca.Function('transition', [self.state.syms,self.input.syms,curvature], [t_integrator])

        # SPATIAL ODE (equations 41a to 41f)
        V_prime = V_dot / s_dot
        s_prime = 1
        ey_prime = ey_dot / s_dot
        epsi_prime = epsi_dot / s_dot
        t_prime = t_dot / s_dot
        state_prime = ca.vertcat(V_prime, s_prime, ey_prime, epsi_prime, t_prime)
        s_ode = ca.Function('ode', [self.state.syms, self.input.syms, curvature], [state_prime])
        s_integrator = integrate(self.state.syms, self.input.syms, curvature, s_ode, h=ds)
        self._spatial_transition = ca.Function('transition', [self.state.syms,self.input.syms,curvature,ds], [s_integrator])

    @property
    def temporal_transition(self):
        return self._temporal_transition
    
    @property
    def spatial_transition(self):
        return self._spatial_transition
    
    
class DynamicPointMassInput(FancyVector):
    def __init__(self, Fx = 0.0, Fy = 0.0):
        """
        :param Fx: longitudinal force | [m/s^2]
        :param Fy: lateral force | [m/s^2]
        """
        self._values = np.array([Fx, Fy])
        self._keys = ['Fx', 'Fy']
        self._syms = ca.vertcat(*[ca.SX.sym(self._keys[i]) for i in range(len(self._keys))])
        
    @property
    def Fx(self): return self.values[0] 
    
    @Fx.setter
    def Fx(self,value: float): 
        assert isinstance(value, float)
        self.values[0] = value
        
    @property
    def Fy(self): return self.values[1] 
    
    @Fx.setter
    def Fy(self,value: float): 
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
    
class DynamicPointMassState(FancyVector):
    def __init__(self, V = 0.0, s = 0.0, ey = 0.0, epsi = 0.0, t = 0.0):
        """
        :param V: longitudinal velocity in global coordinate system | [m/s]
        :param s: curvilinear abscissa | [m]
        :param ey: orthogonal deviation from center-line | [m]
        :param epsi: yaw angle relative to path | [rad]
        :param t: time | [s]
        """
        self._values = np.array([V,s,ey,epsi,t])
        self._keys = ['V','s','ey','epsi','t']
        self._syms = ca.vertcat(*[ca.SX.sym(self._keys[i]) for i in range(len(self._keys))])
    
    @property
    def V(self): return self.values[0] 
    
    @property
    def s(self): return self.values[1]
    
    @property
    def ey(self): return self.values[2]
    
    @ey.setter
    def ey(self, value): self.values[2] = value
    
    @property
    def epsi(self): return self.values[3]
    
    @epsi.setter
    def epsi(self, value): self.values[3] = value
    
    @property
    def t(self): return self.values[4]
    
    @property
    def values(self): return self._values
    
    @property
    def syms(self): return self._syms
    
    @property
    def keys(self): return self._keys
    
    @classmethod
    def create(cls, *args, **kwargs):
        return cls(*args, **kwargs)
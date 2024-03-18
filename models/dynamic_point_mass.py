import casadi as ca
from models.racing_car import RacingCar
from utils.fancy_vector import FancyVector
from utils.common_utils import *
from collections import namedtuple
from casadi import cos, sin, tan, atan, fabs, sign, tanh

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
    
    # TODO: if there is some value to print, do it here
    def print(self, state, input):
        V,s,ey,epsi,t = state
        Fx, Fy = input
        print("##########################")
        
        
    def _init_model(self):
        car = self.config['car']
        env = self.config['env']
        
        # =========== State and auxiliary variables ===================================
        V,s,ey,epsi,t = self.state.variables
        curvature = ca.SX.sym('curvature')
        ds = ca.SX.sym('ds')
        g = 9.88
        
        # ====== Input Model ==========================================================
        Fx, Fy = self.input.variables
        Xd = car['Xd'] # drive distribution
        Xb = car['Xb'] # brake distribution
        
        Xf = (Xd['f']-Xb['f'])/2 * tanh(2*(Fx/1000 + 0.5)) + (Xd['f'] + Xb['f'])/2
        self.Xf = ca.Function("Xf",[Fx],[Xf])
        Fx_f = Fx*Xf
        self.Fx_f = ca.Function("Fx_f",[Fx],[Fx_f])
        
        Xr = (Xb['r']-Xd['r'])/2 * tanh(-2*(Fx/1000 + 0.5)) + (Xd['r'] + Xb['r'])/2
        self.Xr = ca.Function("Xr",[Fx],[Xr])
        Fx_r = Fx*Xr
        self.Fx_r = ca.Function("Fx_r",[Fx],[Fx_r]) 
        
        # ================= Normal Load ================================================
        Fz_f = (car['b']/car['l'])*car['m']*(g*cos(env['theta'])*cos(env['phi']) + env['Av2']*V**2) - car['h']*Fx/car['l']
        self.Fz_f = ca.Function("Fz_f",[V,Fx],[Fz_f])
        
        Fz_r = (car['a']/car['l'])*car['m']*(g*cos(env['theta'])*cos(env['phi']) + env['Av2']*V**2) + car['h']*Fx/car['l']
        self.Fz_r = ca.Function("Fz_f",[V,Fx],[Fz_r])
        
        # ===================== Differential Equations ===================================
        Fb = 0 #-p.m*g*ca.cos(theta)*ca.sin(phi) TODO if you want to change the angle modify this
        Fn = -car['m']*g #ca.cos(theta) is 1 for theta=0, might aswell not write it
        Frr = env['Frr'] #env['Crr']*Fn #rolling resistance = coefficient*normal force (not specified in the paper)
        Fd = Frr + env['Cd']*(V**2) #p.m*g*ca.sin(theta)

        # TEMPORAL transition (equations 1a to 1f)
        V_dot = (Fx - Fd)/car['m']
        s_dot = (V*ca.cos(epsi))/(1-curvature*ey)
        ey_dot = V*ca.sin(epsi)
        epsi_dot = (Fy + Fb)/(car['m']*V) - curvature*s_dot
        t_dot = 1
        state_dot = ca.vertcat(V_dot,  s_dot, ey_dot, epsi_dot, t_dot)
        t_ode = ca.Function('ode', [self.state.syms,self.input.syms,curvature], [state_dot])
        t_integrator = self.integrate(self.state.syms,self.input.syms,curvature,t_ode,self.dt)
        self._temporal_transition = ca.Function('transition', [self.state.syms,self.input.syms,curvature], [t_integrator])

        # SPATIAL transition (equations 41a to 41f)
        V_prime = V_dot / s_dot
        s_prime = 1
        ey_prime = ey_dot / s_dot
        epsi_prime = epsi_dot / s_dot
        t_prime = t_dot / s_dot
        state_prime = ca.vertcat(V_prime, s_prime, ey_prime, epsi_prime, t_prime)
        s_ode = ca.Function('ode', [self.state.syms, self.input.syms, curvature], [state_prime])
        s_integrator = self.integrate(self.state.syms, self.input.syms, curvature, s_ode, h=ds)
        self._spatial_transition = ca.Function('transition', [self.state.syms,self.input.syms,curvature,ds], [s_integrator])

    @property
    def transition(self):
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
        self._labels = [r'$F_x$',r'$F_y$']
        
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
        self.delta = 0 #fictituous steering angle (always zero)
    
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
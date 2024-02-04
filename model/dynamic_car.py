import casadi as ca
from model.racing_car import RacingCar
from utils.fancy_vector import FancyVector
from utils.common_utils import *
from casadi import cos, sin, tan, sqrt, atan, fabs, sign, tanh

class DynamicCar(RacingCar):
    
    @classmethod
    def create_state(cls, *args, **kwargs):
        return DynamicCarState(*args, **kwargs)
    
    @classmethod
    def create_input(cls, *args, **kwargs):
        return DynamicCarInput(*args, **kwargs)
    
    def _init_model(self):
        
        # =========== State and auxiliary variables ===================================
        Ux,Uy,r,delta,s,ey,epsi,t = self.state.variables
        curvature = ca.SX.sym('curvature')
        ds = ca.SX.sym('ds')
        g =  9.88 # gravity
        eps = 0.85 #from the paper
        car = self.config['car']
        env = self.config['env']
        
        # ====== Input Model (Fx needs to be distributed between front and rear) ======
        Fx,w = self.input.variables
        Xd = car['Xd'] # drive distribution
        Xb = car['Xb'] # brake distribution
        
        Xf = (Xd['f']-Xb['f'])/2 * tanh(2*(Fx + 0.5)) + (Xd['f'] + Xb['f'])/2
        self.Xf = ca.Function("Xf",[Fx],[Xf])
        Fx_f = Fx*Xf
        self.Fx_f = ca.Function("Fx_f",[Fx],[Fx_f])
        
        Xr = (Xb['r']-Xd['r'])/2 * tanh(-2*(Fx + 0.5)) + (Xd['r'] + Xb['r'])/2
        self.Xr = ca.Function("Xr",[Fx],[Xr])
        Fx_r = Fx*Xr
        self.Fx_r = ca.Function("Fx_f",[Fx],[Fx_r])
        
        # ================= Normal Load ================================================
        Fz_f = car['b']/self.length*car['m']*(g*cos(env['theta'])*cos(env['phi']) + env['Av2']*Ux**2) - car['h']*Fx/self.length
        self.Fz_f = ca.Function("Fz_f",[Ux,Fx],[Fz_f])
        
        Fz_r = car['a']/self.length*car['m']*(g*cos(env['theta'])*cos(env['phi']) + env['Av2']*Ux**2) + car['h']*Fx/self.length
        self.Fz_r = ca.Function("Fz_f",[Ux,Fx],[Fz_r])
        
        # ================ Maximum Lateral Tire Force ==================================
        Fymax_f = (env['mu']['f']*Fz_f)**2 - ((0.98*Fx_f)**2) # TODO sqrt?
        
        Fymax_r = (env['mu']['r']*Fz_r)**2 - ((0.98*Fx_r)**2) # TODO sqrt?
        
        # ================ Slip Angles equations 11a/b =================================
        alpha_f = atan((Uy + car['a'] * r) / Ux) - delta
        self.alpha_f = ca.Function("alpha_f",[Ux,Uy,r,delta],[alpha_f])
        
        alpha_r = atan((Uy - car['b'] * r) / Ux)
        self.alpha_r = ca.Function("alpha_r",[Ux,Uy,r,delta],[alpha_r])
        
        # ================ Lateral Force ===============================================
        Calpha_f = env['C_alpha']['f']
        alphamod_f = atan(3*Fymax_f*eps/Calpha_f)
        self.alphamod_f = ca.Function("alphamod_f",[Fx],[alphamod_f])
        Fy_f = ca.if_else((ca.fabs(alpha_f) <= alphamod_f),
            -Calpha_f*tan(alpha_f) + Calpha_f**2*fabs(tan(alpha_f))*tan(alpha_f) / (3*Fymax_f) - \
                        (Calpha_f**3*tan(alpha_f)**3)/(27*Fymax_f**2),
            -Calpha_f*(1 - 2*eps + eps**2)*tan(alpha_f) - Fymax_f*(3*eps**2 - 2*eps**3)*sign(alpha_f))
        self.Fy_f = ca.Function("Fy_f",[Ux,Uy,r,delta,Fx],[Fy_f])
        
        Calpha_r = env['C_alpha']['r']
        alphamod_r = atan(3*Fymax_r*eps/Calpha_r)
        self.alphamod_r = ca.Function("alphamod_r",[Fx],[alphamod_r])
        Fy_r = ca.if_else((ca.fabs(alpha_r) <= alphamod_r),
            -Calpha_r*tan(alpha_r) + Calpha_r**2*fabs(tan(alpha_r))*tan(alpha_r) / (3*Fymax_r) - \
                        (Calpha_r**3*tan(alpha_r)**3)/(27*Fymax_r**2),
            -Calpha_r*(1 - 2*eps + eps**2)*tan(alpha_r) - Fymax_r*(3*eps**2 - 2*eps**3)*sign(alpha_r))
        self.Fy_r = ca.Function("Fy_r",[Ux,Uy,r,delta,Fx],[Fy_r])
        
        # ===================== Differential Equations ===================================
        Fb = 0 #-p.m*g*ca.cos(theta)*ca.sin(phi) TODO if you want to change the angle modify this
        Fn = car['m']*g #ca.cos(theta) is 1 for theta=0, might aswell not write it
        Frr = 280 #env['Crr']*Fn #rolling resistance = coefficient*normal force (not specified in the paper)
        Fd = Frr + env['Cd']*(Ux**2) #p.m*g*ca.sin(theta) 

        # TEMPORAL Transition (equations 1a to 1f)
        Ux_dot = (Fx_f*cos(delta) - Fy_f*sin(delta) + Fx_r - Fd)/car['m'] + r*Uy
        Uy_dot = (Fy_f*cos(delta) + Fx_f*sin(delta) + Fy_r + Fb)/car['m'] - r*Ux
        r_dot = (car['a']*(Fy_f*cos(delta) + Fx_f*sin(delta)) - car['b']*Fy_r) / car['Izz']
        delta_dot = w 
        s_dot = (Ux*cos(epsi) - Uy*sin(epsi)) / (1 - curvature*ey)
        ey_dot = Ux*sin(epsi) + Uy*cos(epsi)
        epsi_dot = r - curvature*s_dot
        t_dot = 1
        state_dot = ca.vertcat(Ux_dot, Uy_dot, r_dot, delta_dot, s_dot, ey_dot, epsi_dot, t_dot)
        t_ode = ca.Function('ode', [self.state.syms,self.input.syms,curvature], [state_dot])
        t_integrator = integrate(self.state.syms,self.input.syms,curvature,t_ode,self.dt)
        self._temporal_transition = ca.Function('transition', [self.state.syms,self.input.syms,curvature], [t_integrator])

        # SPATIAL Transition (equations 41a to 41f)
        Ux_prime = Ux_dot / s_dot
        Uy_prime = Uy_dot / s_dot
        r_prime = r_dot / s_dot
        delta_prime = delta_dot / s_dot
        s_prime = 1
        ey_prime = ey_dot / s_dot
        epsi_prime = epsi_dot / s_dot
        t_prime = t_dot / s_dot
        state_prime = ca.vertcat(Ux_prime, Uy_prime, r_prime, delta_prime, s_prime, ey_prime, epsi_prime, t_prime)
        s_ode = ca.Function('ode', [self.state.syms, self.input.syms, curvature], [state_prime])
        s_integrator = integrate(self.state.syms, self.input.syms, curvature, s_ode, h=ds)
        self._spatial_transition = ca.Function('transition', [self.state.syms,self.input.syms,curvature,ds], [s_integrator])
    
    @property
    def transition(self):
        return self._temporal_transition
    
    @property
    def spatial_transition(self):
        return self._spatial_transition
    
class DynamicCarInput(FancyVector):
    def __init__(self, Fx = 0.0, w = 0.0):
        """
        :param Fx: longitudinal force | [m/s^2]
        :param w: steering angle rate | [rad/s]
        """
        self._values = np.array([Fx,w])
        self._keys = ['Fx','w']
        self._syms = ca.vertcat(*[ca.SX.sym(self._keys[i]) for i in range(len(self._keys))])
        
    @property
    def Fx(self): return self.values[0] 
      
    @property
    def w(self): return self.values[1]
    
    @Fx.setter
    def Fx(self,value: float): 
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
    
class DynamicCarState(FancyVector):
    def __init__(self, Ux = 0.0, Uy = 0.0, r = 0.0, delta = 0.0, s = 0.0, ey = 0.0, epsi = 0.0, t = 0.0):
        """
        :param Ux: longitudinal velocity in global coordinate system | [m/s]
        :param Uy: lateral velocity in global coordinate system | [m/s]
        :param r: yaw rate | [rad/s]
        :param delta: steering angle | [rad]
        :param s: curvilinear abscissa | [m]
        :param ey: orthogonal deviation from center-line | [m]
        :param epsi: yaw angle relative to path | [rad]
        :param t: time | [s]
        """
        self._values = np.array([Ux,Uy,r,delta,s,ey,epsi,t])
        self._keys = ['Ux','Uy','r','delta','s','ey','epsi','t']
        self._syms = ca.vertcat(*[ca.SX.sym(self._keys[i]) for i in range(len(self._keys))])
    
    @property
    def Ux(self): return self.values[0] 
    
    @property
    def Uy(self): return self.values[1] 
    
    @property
    def r(self): return self.values[2]
    
    @property
    def delta(self): return self.values[3]
      
    @property
    def s(self): return self.values[4]
    
    @property
    def ey(self): return self.values[5]
    
    @ey.setter
    def ey(self, value): self.values[5] = value
    
    @property
    def epsi(self): return self.values[6]
    
    @epsi.setter
    def epsi(self, value): self.values[6] = value
    
    @property
    def t(self): return self.values[7]
    
    @property
    def values(self): return self._values
    
    @property
    def syms(self): return self._syms
    
    @property
    def keys(self): return self._keys

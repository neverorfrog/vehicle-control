import casadi as ca
from vehicle_control.models.racing_car import RacingCar
from vehicle_control.utils.fancy_vector import FancyVector
from vehicle_control.utils.common_utils import *
from casadi import cos, sin, tan, atan, fabs, sign, tanh
from vehicle_control.utils.integrators import Euler, RK4, RK2

class DynamicCar(RacingCar):
    
    @classmethod
    def create_state(cls, *args, **kwargs):
        return DynamicCarState(*args, **kwargs)
    
    @classmethod
    def create_action(cls, *args, **kwargs):
        return DynamicCarAction(*args, **kwargs)
    
    def print(self, state, input):
        Ux, Uy, r, delta, s, ey, epsi, t = state
        Fx, w = input
        env = self.config.env
        muf = env.mu.f; mur = env.mu.r
        print(f"Fx_f: {self.Fx_f(Fx)}")
        print(f"Fx_r: {self.Fx_r(Fx)}")
        print(f"Fy_f: {self.Fy_f(Ux,Uy,r,delta,Fx)}")
        print(f"Fy_r: {self.Fy_r(Ux,Uy,r,delta,Fx)}")
        Ux_dot = self.Ux_dot(Fx,Ux,Uy,r,delta).full().squeeze().item()
        Uy_dot = self.Uy_dot(Fx,Ux,Uy,r,delta).full().squeeze().item()
        print(f"UX ACCELERATION: {Ux_dot:.3f}, UY ACCELERATION: {Uy_dot:.3f}")
        Fymax_f = ((muf*self.Fz_f(Ux,Fx))**2 - ((0.99*self.Fx_f(Fx))**2))
        Fymax_r = ((mur*self.Fz_r(Ux,Fx))**2 - ((0.99*self.Fx_r(Fx))**2))
        alpha_f = np.rad2deg(self.alpha_f(Ux,Uy,r,delta).full().squeeze().item())
        alpha_r = np.rad2deg(self.alpha_r(Ux,Uy,r,delta).full().squeeze().item())
        alphamod_f = np.rad2deg(self.alphamod_f(Fx).full().squeeze().item())
        alphamod_r = np.rad2deg(self.alphamod_r(Fx).full().squeeze().item())
        slipping_f = ca.fabs(alpha_f) - ca.fabs(alphamod_f)
        slipping_r = ca.fabs(alpha_r) - ca.fabs(alphamod_r)
        if slipping_f > 0 or slipping_r > 0:
            print(f"alpha_f: {alpha_f:.2f}")
            print(f"alpha_r: {alpha_r:.2f}")
            print(f"slipping_f: {slipping_f:.2f}")
            print(f"slipping_r: {slipping_r:.2f}")
            print(f"Fy_f diff: {self.Fy_f(Ux,Uy,r,delta, Fx)**2 - Fymax_f}")
            print(f"Fy_r diff: {self.Fy_r(Ux,Uy,r,delta, Fx)**2 - Fymax_r}")
    
    def _init_model(self):
        # =========== State and auxiliary variables ===================================
        Ux,Uy,r,delta,s,ey,epsi,t = self.state.variables
        self.Ux = Ux
        self.Uy = Uy
        self.r = r
        self.delta = delta
        self.ey = ey
        self.epsi = epsi
        curvature = ca.SX.sym('curvature')
        dt = ca.SX.sym('dt')
        ds = ca.SX.sym('ds')
        g =  9.88 # gravity
        car = self.config.car
        env = self.config.env
        eps = car.eps
        
        # ====== Input Model (Fx needs to be distributed between front and rear) ======
        Fx,w = self.input.variables
        self.Fx = Fx
        Xd = car.Xd; Xdf = Xd.f; Xdr = Xd.r; # drive distribution
        Xb = car.Xb; Xbf = Xb.f; Xbr = Xb.r; # brake distribution
         
        Xf = (Xdf - Xbf)/2 * tanh(2*(Fx/1000 + 0.5)) + (Xdf + Xbf)/2
        self.Xf = ca.Function("Xf",[Fx],[Xf]).expand()
        Fx_f = Fx*Xf
        self.Fx_f = ca.Function("Fx_f",[Fx],[Fx_f]).expand()
        
        Xr = (Xbr-Xdr)/2 * tanh(-2*(Fx/1000 + 0.5)) + (Xdr + Xbr)/2
        self.Xr = ca.Function("Xr",[Fx],[Xr]).expand()
        Fx_r = Fx*Xr
        self.Fx_r = ca.Function("Fx_r",[Fx],[Fx_r]).expand()
        
        # ================= Normal Load ================================================
        a = car.a; b = car.b; l = car.l; m = car.m; h = car.h
        theta = env.theta; phi = env.phi; Av2 = env.Av2
        
        Fz_f = (b/l)*m*(g*cos(theta)*cos(phi) + Av2*Ux**2) - h*Fx/l
        self.Fz_f = ca.Function("Fz_f",[Ux,Fx],[Fz_f]).expand()
        
        Fz_r = (a/l)*m*(g*cos(theta)*cos(phi) + Av2*Ux**2) + h*Fx/l
        self.Fz_r = ca.Function("Fz_f",[Ux,Fx],[Fz_r]).expand()
        
        # ================ Maximum Lateral Tire Force ==================================
        muf = env.mu.f; mur = env.mu.r
        Fymax_f = ((muf*Fz_f)**2 - ((0.99*Fx_f)**2))**0.5
        Fymax_r = ((mur*Fz_r)**2 - ((0.99*Fx_r)**2))**0.5
        
        # ================ Slip Angles equations 11a/b =================================
        alpha_f = atan((Uy + a * r) / Ux) - delta
        self.alpha_f = ca.Function("alpha_f",[Ux,Uy,r,delta],[alpha_f]).expand()
        
        alpha_r = atan((Uy - b * r) / Ux)
        self.alpha_r = ca.Function("alpha_r",[Ux,Uy,r,delta],[alpha_r]).expand()
        
        # ================ Lateral Force ===============================================
        Calpha_f = car.C_alpha.f
        alphamod_f = atan((3*Fymax_f*eps)/Calpha_f)
        self.alphamod_f = ca.Function("alphamod_f",[Fx],[alphamod_f]).expand()
        Fy_f = ca.if_else((ca.fabs(alpha_f) <= alphamod_f),
            -Calpha_f*tan(alpha_f) + Calpha_f**2*fabs(tan(alpha_f))*tan(alpha_f) / (3*Fymax_f) - \
                (Calpha_f**3*tan(alpha_f)**3)/(27*Fymax_f**2), #first case, 
            -Calpha_f*(1 - 2*eps + eps**2)*tan(alpha_f) - Fymax_f*(3*eps**2 - 2*eps**3)*sign(alpha_f))
        self.Fy_f = ca.Function("Fy_f",[Ux,Uy,r,delta,Fx],[Fy_f]).expand()
        
        Calpha_r = car.C_alpha.r
        alphamod_r = atan((3*Fymax_r*eps)/Calpha_r)
        self.alphamod_r = ca.Function("alphamod_r",[Fx],[alphamod_r]).expand()
        Fy_r = ca.if_else((ca.fabs(alpha_r) <= alphamod_r),
            -Calpha_r*tan(alpha_r) + Calpha_r**2*fabs(tan(alpha_r))*tan(alpha_r) / (3*Fymax_r) - \
                        (Calpha_r**3*tan(alpha_r)**3)/(27*Fymax_r**2),
            -Calpha_r*(1 - 2*eps + eps**2)*tan(alpha_r) - Fymax_r*(3*eps**2 - 2*eps**3)*sign(alpha_r))
        self.Fy_r = ca.Function("Fy_r",[Ux,Uy,r,delta,Fx],[Fy_r]).expand()
        
        # ===================== Differential Equations ===================================
        Fb = 0 #-p.m*g*ca.cos(theta)*ca.sin(phi) TODO if you want to change the angle modify this
        Fn = -m*g #ca.cos(theta) is 1 for theta=0, might aswell not write it
        Frr = env.Frr #env['Crr']*Fn #rolling resistance = coefficient*normal force (not specified in the paper)
        Fd = Frr + env.Cd*(Ux**2) #p.m*g*ca.sin(theta) 

        Izz = car.Izz
        # TEMPORAL Transition (equations 1a to 1f)
        Ux_dot = (Fx_f*cos(delta) - Fy_f*sin(delta) + Fx_r - Fd)/m + r*Uy
        Uy_dot = (Fy_f*cos(delta) + Fx_f*sin(delta) + Fy_r + Fb)/m - r*Ux
        r_dot = (a*(Fy_f*cos(delta) + Fx_f*sin(delta)) - b*Fy_r) / Izz
        delta_dot = w 
        s_dot = (Ux*cos(epsi) - Uy*sin(epsi)) / (1 - curvature*ey)
        ey_dot = Ux*sin(epsi) + Uy*cos(epsi)
        epsi_dot = r - curvature*s_dot
        t_dot = 1
        state_dot = ca.vertcat(Ux_dot, Uy_dot, r_dot, delta_dot, s_dot, ey_dot, epsi_dot, t_dot)
        self.Ux_dot = ca.Function("Ux_dot",[Fx,Ux,Uy,r,delta],[Ux_dot]).expand()
        self.Uy_dot = ca.Function("Uy_dot",[Fx,Ux,Uy,r,delta],[Uy_dot]).expand()
        time_integrator = RK4(self.state.syms, self.input.syms, curvature, state_dot, dt)
        self._temporal_transition = time_integrator.step

        # SPATIAL Transition (equations 41a to 41f)
        Ux_prime = Ux_dot / s_dot
        Uy_prime = Uy_dot / s_dot
        r_prime = r_dot / s_dot
        delta_prime = w / s_dot
        s_prime = 1
        ey_prime = ey_dot / s_dot
        epsi_prime = epsi_dot / s_dot
        t_prime = 1 / s_dot
        state_prime = ca.vertcat(Ux_prime, Uy_prime, r_prime, delta_prime, s_prime, ey_prime, epsi_prime, t_prime)
        space_integrator = RK4(self.state.syms, self.input.syms, curvature, state_prime, ds)
        self._spatial_transition = space_integrator.step
    
    @property
    def transition(self):
        return self._temporal_transition
    
    @property
    def spatial_transition(self):
        return self._spatial_transition
    
class DynamicCarAction(FancyVector):
    def __init__(self, Fx = 0.0, w = 0.0):
        """
        :param Fx: longitudinal force | [N]
        :param w: steering angle rate | [rad/s]
        """
        self._values = np.array([Fx,w])
        self._keys = ['Fx','w']
        self._syms = ca.vertcat(*[ca.SX.sym(self._keys[i]) for i in range(len(self._keys))])
        self._labels = [r'$F_x$',r'$\omega$']
        
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

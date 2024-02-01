import casadi as ca
from model.racing_car import RacingCar
from utils.fancy_vector import FancyVector
from utils.common_utils import *
from collections import namedtuple

class DynamicCar(RacingCar):
    
    @classmethod
    def create_state(cls, *args, **kwargs):
        return DynamicCarState(*args, **kwargs)
    
    @classmethod
    def create_input(cls, *args, **kwargs):
        return DynamicCarInput(*args, **kwargs)
    
    def _init_ode(self):
        # Input variables
        Fx,w = self.input.variables

        # State and auxiliary variables
        Ux,Uy,r,delta,s,ey,epsi,t = self.state.variables
        curvature = ca.SX.sym('curvature')
        ds = ca.SX.sym('ds')

        #parameters: mass, distance of CG from front (a) and rear (b) axis, height of CG, yaw moment of inertia
        p = self.get_car_parameters()

        g,theta,phi,Av2,Crr,eps,mu,C_alpha,Cd = self.get_parameters()
        
        #utils for ODE
        Xf, Xr = self._get_force_distribution(Fx)

        Fb = 0 #-p.m*g*ca.cos(theta)*ca.sin(phi) TODO if you want to change the angle modify this
        Fn = -p.m*g*1 #ca.cos(theta) is 1 for theta=0, might aswell not write it
        
        Frr = Crr*Fn #rolling resistance = coefficient*normal force (not specified in the paper)
        
        #All the forces introduced above are constant, as the various coefficient are constant and the ground is always flat
        #Fd depends on the velocity instead, so we define a casadi function (Is this the correct methodology?)
        Fd = ca.Function("Fd_Function",[Ux], [Frr + Cd*Ux**2 - 0]) #p.m*g*ca.sin(theta) 
        
        #Fz is also not constant, we define 2 casadi functions for front and rear (15a/b in the paper)
        #TODO Fx here should be just the Fx value? or Fx*Xf when we're computing Fz_f and Fx*Xr when computing Fz_r?
        L = (p.a+p.b)
        Fz_f = self.get_Fz_f_function(Ux, Fx, Xf, p, L, g, theta, phi, Av2)
        Fz_r = self.get_Fz_r_function(Ux, Fx, Xf, p, L, g, theta, phi, Av2)

        #slip angles, equations 11a/b
        alpha_f = self.get_alpha_f_function(Uy, Ux, delta, r, p) 
        alpha_r = self.get_alpha_r_function(Uy, Ux, r, p)
        
        #Fy is rather complex to obtain, self.get_lateral_force returns:
        #a casadi function mapping (Uy, Ux, delta, r, Fx) to [Fy_f, Fy_r]
        Fy = self._get_lateral_force(alpha_f, alpha_r,Uy, Ux, delta, r ,Fz_f, Fz_r, Fx, Xf, eps, mu, C_alpha)
        
        # TEMPORAL ODE (equations 1a to 1f)
        Ux_dot = (Fx*Xf(Fx)*ca.cos(delta) - Fy(Uy, Ux, delta, r, Fx)[0] * ca.sin(delta) + Fx*Xr(Fx) - Fd(Ux))/p.m + r*Uy
        Uy_dot = (Fy(Uy, Ux, delta, r, Fx)[0] * ca.cos(delta) + Fx*Xf(Fx)*ca.sin(delta) + Fy(Uy, Ux, delta, r, Fx)[1] + Fb)/p.m - r*Ux
        r_dot = (p.a*(Fy(Uy, Ux, delta, r, Fx)[0]*ca.cos(delta) + Fx*Xf(Fx)*ca.sin(delta)) - p.b*Fy(Uy, Ux, delta, r, Fx)[1]) / p.Izz #TODO moment of inertia? maybe from here? http://archive.sciendo.com/MECDC/mecdc.2013.11.issue-1/mecdc-2013-0003/mecdc-2013-0003.pdf
        delta_dot = w 
        s_dot = (Ux*ca.cos(epsi) - Uy*ca.sin(epsi)) / (1 - curvature*ey)
        ey_dot = Ux*ca.sin(epsi) + Uy*ca.cos(epsi)
        epsi_dot = r - curvature*s_dot
        t_dot = 1
        state_dot = ca.vertcat(Ux_dot, Uy_dot, r_dot, delta_dot, s_dot, ey_dot, epsi_dot, t_dot)
        t_ode = ca.Function('ode', [self.state.syms,self.input.syms,curvature], [state_dot])
        t_integrator = integrate(self.state.syms,self.input.syms,curvature,t_ode,self.dt)
        self._temporal_transition = ca.Function('transition', [self.state.syms,self.input.syms,curvature], [t_integrator])

        # SPATIAL ODE (equations 41a to 41f)
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

    def _get_force_distribution(self, Fx):
        """
        returns two casadi functions, to get the front/rear force distribution given Fx
        equation 6a/6b in the paper
        """
        #for rear only drive:
        df = 0
        dr = 1
        bf = 0.5
        br = 0.5 
        Xf = ca.Function("force_distribution", [Fx], [(df-bf)/2 * ca.tanh(2*(Fx + 0.5)) + (df + bf)/2])
        Xr = ca.Function("force_distribution", [Fx], [(br-dr)/2 * ca.tanh(-2*(Fx + 0.5)) + (dr + br)/2 ])
        return Xf, Xr

    def _get_lateral_force(self, alpha_f, alpha_r, Uy, Ux, delta, r, Fz_f, Fz_r, Fx, Xf, eps, mu, C_alpha):
        
        #Fy_max is 2 DIMENSIONAL, has front and rear values
        Fy_max = self.get_Fy_max_function(Ux, Fx, Fz_f, Fz_r, Xf, mu)

        alpha_mod = self.get_alpha_mod_function(Ux, Fx, Fy_max, eps, C_alpha)

        condition = ca.logic_and((ca.fabs(alpha_f(Uy, Ux, delta, r)) <= alpha_mod(Ux, Fx)[0]) , (ca.fabs(alpha_r(Uy, Ux, r)) <= alpha_mod(Ux, Fx)[1]))

        # Element-wise operations based on conditions
        result = ca.if_else(
            condition,
            ca.vertcat(
            -C_alpha*ca.tan(alpha_f(Uy, Ux, delta, r)) + C_alpha**2*ca.fabs(ca.tan(alpha_f(Uy, Ux, delta, r)))*ca.tan(alpha_f(Uy, Ux, delta, r)) / (3*Fy_max(Ux, Fx)[0]) - (C_alpha**3*ca.tan(alpha_f(Uy, Ux, delta, r))**3)/(27*Fy_max(Ux, Fx)[0]**2),
            -C_alpha*ca.tan(alpha_r(Uy, Ux, r)) + C_alpha**2*ca.fabs(ca.tan(alpha_r(Uy, Ux, r)))*ca.tan(alpha_r(Uy, Ux, r)) / (3*Fy_max(Ux, Fx)[1]) - (C_alpha**3*ca.tan(alpha_r(Uy, Ux, r))**3)/(27*Fy_max(Ux, Fx)[1]**2)
            ),
            ca.vertcat(
            -C_alpha*(1 - 2*eps + eps**2)*ca.tan(alpha_f(Uy, Ux, delta, r)) - Fy_max(Ux, Fx)[0]*(3*eps**2 - 2*eps**3)*ca.sign(alpha_f(Uy, Ux, delta, r)),
            -C_alpha*(1 - 2*eps + eps**2)*ca.tan(alpha_r(Uy, Ux, r)) - Fy_max(Ux, Fx)[1]*(3*eps**2 - 2*eps**3)*ca.sign(alpha_r(Uy, Ux, r))
            )
        )
        return ca.Function("lateral_forces", [Uy, Ux, delta, r, Fx], [result])
    
    @property
    def temporal_transition(self):
        return self._temporal_transition
    
    @property
    def spatial_transition(self):
        return self._spatial_transition
    
    def get_alpha_f_function(self, Uy, Ux, delta, r, p):
        alpha_f_function = ca.Function("alpha_f_Function", [Uy, Ux, delta, r], [ca.atan((Uy + p.a * r) / Ux) - delta])
        return alpha_f_function
    
    def get_alpha_r_function(self, Uy, Ux, r, p):
        alpha_r_function = ca.Function("alpha_r_Function", [Uy, Ux, r], [ca.atan((Uy - p.b*r)/Ux)])
        return alpha_r_function
    
    def get_alpha_mod_function(self, Ux, Fx, Fy_max, eps, C_alpha):
        alpha_mod_function = ca.Function("alpha_mod_Function", [Ux, Fx], [ca.atan(3*Fy_max(Ux,Fx)*eps/C_alpha)]) #alpha mod is 2D because Fymax is 2D because Fz is 2D
        return alpha_mod_function
    
    def get_Fy_max_function(self, Ux, Fx, Fz_f, Fz_r, Xf, mu):
        Fy_max_function = ca.Function("Fy_max_Function", [Ux, Fx], [((mu*ca.vertcat(Fz_f(Ux, Fx), Fz_r(Ux, Fx)))**2 - (0.99*ca.vertcat(Fx*Xf(Fx), Fx*Xf(Fx)))**2)**0.5 ]) #TODO should Fx also be vertcat(Fx_f)
        return Fy_max_function
    
    def get_Fz_f_function(self, Ux, Fx, Xf, p, L, g, theta, phi, Av2):
        Fz_f_function = ca.Function("Fz_f_Function", [Ux, Fx], [p.b/L * p.m*(g*ca.cos(theta)*ca.cos(phi) + Av2*Ux**2) - p.h_cg*Fx*Xf(Fx)/L])
        return Fz_f_function
    
    def get_Fz_r_function(self, Ux, Fx, Xf, p, L, g, theta, phi, Av2):
        Fz_r_function = ca.Function("Fz_r_Function", [Ux, Fx], [p.a/L*p.m*(g*ca.cos(theta)*ca.cos(phi)+Av2*Ux**2) + p.h_cg*Fx*Xf(Fx)/L])
        return Fz_r_function
    
    def get_car_parameters(self):
        Parameters = namedtuple('Parameters', ['m', 'a', 'b', 'h_cg', 'Izz'])
        p = Parameters(1778, 1.194, 1.436, 0.55, 3049)
        return p
    
    def get_parameters(self):
        g = 9.88

        #TODO theta and phi are road grade and bank angle, but for now we assume flat track
        theta = 0 
        phi = 0

        Av2 = 0                 #because theta, phi are 0
        Crr = 0.014             #https://en.wikipedia.org/wiki/Rolling_resistance
        eps = 0.85              #from the paper
        mu = 0.4                #http://hyperphysics.phy-astr.gsu.edu/hbase/Mechanics/frictire.htmlFx
        C_alpha = 3.5           #https://www.politesi.polimi.it/bitstream/10589/152539/3/2019_12_Viani.pdf  proposes some real value taken from wherever, we should look at this in more detail
        Cd = 0.25               #https://en.wikipedia.org/wiki/Automobile_drag_coefficient#:~:text=The%20average%20modern%20automobile%20achieves,a%20Cd%3D0.35%E2%80%930.45.

        return g,theta,phi,Av2,Crr,eps,mu,C_alpha,Cd
    
    
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
    
    @classmethod
    def create(cls, *args, **kwargs):
        return cls(*args, **kwargs)
    
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
    
    @classmethod
    def create(cls, *args, **kwargs):
        return cls(*args, **kwargs)
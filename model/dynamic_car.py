import casadi as ca
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from environment.track import Track, Waypoint
from model.state import DynamicCarInput, DynamicCarState
from utils.utils import *
from collections import namedtuple
from casadi import cos, sin, tan

class DynamicCar():
    def __init__(self, track: Track, length, dt):
        """
        Dynamic Bicycle Model
        :param track: reference path object to follow
        :param length: length of car in m
        :param dt: sampling time of model
        """
        # Precision
        self.eps = 1e-12
        # Car Parameters
        self.length = length
        # Reference Path
        self.track = track
        # Set sampling time
        self.dt = dt
        
        # Initialize state
        self.wp_id = 0
        self.current_waypoint: Waypoint = self.track.waypoints[self.wp_id]
        self.state: DynamicCarState = DynamicCarState()
        
        # Initialize input (fictituous)
        self.input: DynamicCarInput = DynamicCarInput()
        
        # Initialize dynamic model
        self._init_ode()
        
        # function for evaluating curvature at given s
        s = ca.MX.sym('s')
        self.curvature = ca.Function("curvature",[s],[self.track.get_curvature(s)])
        
    def drive(self, input: DynamicCarInput):
        """
        :param input: input vector containing [a, w]
        """
        curvature = self.track.get_curvature(self.state.s)
        next_state = self.temporal_transition(self.state.values, input.values, curvature).full().squeeze()
        self.state = DynamicCarState(*next_state)
        self.input = input
        return self.state
    
    def rel2glob(self, state):
        s = state[self.state.index('s')]  
        ey = state[self.state.index('ey')] 
        epsi = state[self.state.index('epsi')]    
        track_psi = wrap(self.track.get_orientation(s))
        x = self.track.x(s) - sin(track_psi) * ey
        y = self.track.y(s) + cos(track_psi) * ey
        psi = track_psi + epsi
        return x.full().squeeze(),y.full().squeeze(),psi.full().squeeze()
        
    
    def _init_ode(self):
        g = 9.88
        # Input variables
        Fx,w = self.input.variables

        # State and auxiliary variables
        Ux,Uy,delta,r,s,ey,epsi,t = self.state.variables
        curvature = ca.SX.sym('curvature')
        ds = ca.SX.sym('ds')

        #parameters: mass, distance of CG from front (a) and rear (b) axis, height of CG, yaw moment of inertia
        Parameters = namedtuple('Parameters', ['m', 'a', 'b', 'h_cg', 'Izz'])
        p = Parameters(1000, 2, 2, 1, 2500)
        
        #utils for ODE
        Xf, Xr = self._get_force_distribution(Fx)

        #TODO theta and phi are road grade and bank angle, but for now we assume flat track
        theta = 0 
        phi = 0
        Av2 = 0 #because theta, phi are 0

        Fb = 0 #-p.m*g*ca.cos(theta)*ca.sin(phi) TODO if you want to change the angle modify this
        Fn = -p.m*g*1 #ca.cos(theta) is 1 for theta=0, might aswell not write it
        
        Crr = 0.014 #https://en.wikipedia.org/wiki/Rolling_resistance
        Frr = Crr*Fn #rolling resistance = coefficient*normal force (not specified in the paper)
        
        
        #All the forces introduced above are constant, as the various coefficient are constant and the ground is always flat
        #Fd depends on the velocity instead, so we define a casadi function (Is this the correct methodology?)
        Cd = 0.25 #https://en.wikipedia.org/wiki/Automobile_drag_coefficient#:~:text=The%20average%20modern%20automobile%20achieves,a%20Cd%3D0.35%E2%80%930.45.
        Fd = ca.Function("Fd_Function",[Ux], [Frr + Cd*Ux**2 - 0]) #p.m*g*ca.sin(theta) 
        
        #Fz is also not constant, we define 2 casadi functions for front and rear (15a/b in the paper)
        #TODO Fx here should be just the Fx value? or Fx*Xf when we're computing Fz_f and Fx*Xr when computing Fz_r?
        L = (p.a+p.b)
        Fz_f = ca.Function("Fz_f_Function", [Ux, Fx], [p.b/L * p.m*(g*ca.cos(theta)*ca.cos(phi) + Av2*Ux**2) - p.h_cg*Fx*Xf(Fx)/L])
        Fz_r = ca.Function("Fz_r_Function", [Ux, Fx], [p.a/L*p.m*(g*ca.cos(theta)*ca.cos(phi)+Av2*Ux**2) + p.h_cg*Fx*Xf(Fx)/L])

        #slip angles, equations 11a/b
        alpha_f = ca.Function("alpha_f_Function",[Uy, Ux, delta, r], [ca.atan((Uy + p.a*r)/Ux) - delta])
        alpha_r = ca.Function("alpha_r_Function", [Uy, Ux, r], [ca.atan((Uy - p.b*r)/Ux)])
        
        #Fy is rather complex to obtain, self.get_lateral_force returns:
        #a casadi function mapping (Uy, Ux, delta, r, Fx) to [Fy_f, Fy_r]
        Fy = self._get_lateral_force(alpha_f, alpha_r,Uy, Ux, delta, r ,Fz_f, Fz_r, Fx, Xf)
        
        # TEMPORAL ODE (equations 1a to 1f)
        Ux_dot = (Fx*Xf(Fx)*ca.cos(delta) - Fy(Uy, Ux, delta, r, Fx)[0] * ca.sin(delta) + Fx*Xr(Fx) - Fd(Ux))/p.m + r*Uy
        Uy_dot = (Fy(Uy, Ux, delta, r, Fx)[0] * ca.cos(delta) + Fx*Xf(Fx)*ca.sin(delta) + Fy(Uy, Ux, delta, r, Fx)[1] + Fb)/p.m - r*Ux
        r_dot = (p.a*(Fy(Uy, Ux, delta, r, Fx)[0]*ca.cos(delta) + Fx*Xf(Fx)*ca.sin(delta)) - p.b*Fy(Uy, Ux, delta, r, Fx)[1]) / p.Izz #TODO moment of inertia? maybe from here? http://archive.sciendo.com/MECDC/mecdc.2013.11.issue-1/mecdc-2013-0003/mecdc-2013-0003.pdf
        delta_dot = w 
        s_dot = (Ux*ca.cos(epsi) - Uy*ca.sin(epsi)) / (1 - curvature*ey) #TODO The path’s curvature κ defines the horizontal path geometry
        ey_dot = Ux*ca.sin(epsi) + Uy*ca.cos(epsi)
        epsi_dot = r - curvature*s_dot
        t_dot = 1
        state_dot = ca.vertcat(Ux_dot, Uy_dot, r_dot, delta_dot, s_dot, ey_dot, epsi_dot, t_dot)
        t_ode = ca.Function('ode', [self.state.syms,self.input.syms,curvature], [state_dot])
        t_integrator = integrate(self.state.syms,self.input.syms,curvature,t_ode,self.dt)
        self.temporal_transition = ca.Function('transition', [self.state.syms,self.input.syms,curvature], [t_integrator])

        # SPATIAL ODE (equations 41a to 41f)
        s_dot = (Ux*ca.cos(epsi) - Uy*ca.sin(epsi)) / (1 - curvature*ey)
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
        self.spatial_transition = ca.Function('transition', [self.state.syms,self.input.syms,curvature,ds], [s_integrator])

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

    def _get_lateral_force(self, alpha_f, alpha_r, Uy, Ux, delta, r, Fz_f, Fz_r, Fx, Xf):
        eps = 0.85 #from the paper

        mu = 0.4 #http://hyperphysics.phy-astr.gsu.edu/hbase/Mechanics/frictire.htmlFx
        Fy_max = ca.Function("Fy_max_Function", [Ux, Fx], [((mu*ca.vertcat(Fz_f(Ux, Fx), Fz_r(Ux, Fx)))**2 - (0.99*ca.vertcat(Fx*Xf(Fx), Fx*Xf(Fx)))**2)**0.5 ]) #TODO should Fx also be vertcat(Fx_f)
        #Fy_max is 2 DIMENSIONAL, has front and rear values
        C_alpha = 3.5 #https://www.politesi.polimi.it/bitstream/10589/152539/3/2019_12_Viani.pdf  proposes some real value taken from wherever, we should look at this in more detail

        alpha_mod = ca.Function("alpha_mod_Function", [Ux, Fx], [ca.atan(3*Fy_max(Ux,Fx)*eps/C_alpha)]) #alpha mod is 2D because Fymax is 2D because Fz is 2D
        

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
    
    def plot(self, axis: Axes, state: DynamicCarState):
        x,y,psi = self.rel2glob(state)
        delta = state.delta
        r = self.length / 2
        
        # Draw the bicycle as a rectangle
        width = self.length
        height = self.length
        angle = wrap(psi-np.pi/2)
        rectangle = plt.Rectangle((x-np.cos(angle)*width/2-np.cos(psi)*2*width/3, y-np.sin(angle)*height/2-np.sin(psi)*2*height/3),
                                  width,height,edgecolor='black',alpha=0.7, angle=np.rad2deg(angle), rotation_point='xy')
        axis.add_patch(rectangle)
        
        # Draw four wheels as rectangles
        wheel_width = self.length / 10
        wheel_height = self.length / 4
        wheel_angle = wrap(psi+delta-np.pi/2)
        wheel_right_front = plt.Rectangle((x+np.cos(angle)*r, y+np.sin(angle)*r),width=wheel_width,height=wheel_height,angle=np.rad2deg(wheel_angle),facecolor='black')
        axis.add_patch(wheel_right_front)
        wheel_left_front = plt.Rectangle((x-np.cos(angle)*r, y-np.sin(angle)*r),width=wheel_width,height=wheel_height,angle=np.rad2deg(wheel_angle),facecolor='black')
        axis.add_patch(wheel_left_front)
        wheel_right_back = plt.Rectangle((x+np.cos(angle)*r-np.cos(psi)*width*0.6, y+np.sin(angle)*r-np.sin(psi)*height*0.6),width=wheel_width,height=wheel_height,angle=np.rad2deg(wheel_angle),facecolor='black')
        axis.add_patch(wheel_right_back)
        wheel_left_back = plt.Rectangle((x-np.cos(angle)*r-np.cos(psi)*width*0.6, y-np.sin(angle)*r-np.sin(psi)*height*0.6),width=wheel_width,height=wheel_height,angle=np.rad2deg(wheel_angle),facecolor='black')
        axis.add_patch(wheel_left_back)
        
        return x,y
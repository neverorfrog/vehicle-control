import casadi as ca
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from modeling.util import *
from collections import namedtuple
#from robot import Robot

class Robot():
    '''
        Defines the ODE
        q (state), u (input):    casadi expression that have been used to define the dynamics qd
        qd (state_dot):          casadi expr defining the rhs of the ode 
    '''
    def __init__(self, q, u, qd):
        self.q = q; self.u = u; self.qd = qd
        self.q_len = q.shape[0]
        self.u_len = u.shape[0]
        self.transition_function = ca.Function('qdot', [q, u], [qd])
        
    def plot(self, axis: Axes, q):
        '''Plots the actual shape of the robot'''
        pass
        
    def RK4(self,dt,integration_steps=10):
        '''
        RK4 integrator
        dt:             integration interval
        N_steps:        number of integration steps per integration interval, default:1
        '''
        h = dt/integration_steps
        current_q = self.q
        
        for _ in range(integration_steps):
            k_1 = self.transition_function(current_q, self.u)
            k_2 = self.transition_function(current_q + (dt/2)*k_1, self.u)
            k_3 = self.transition_function(current_q + (dt/2)*k_2, self.u)
            k_4 = self.transition_function(current_q + dt*k_3, self.u)
            current_q += (1/6) * (k_1 + 2 * k_2 + 2 * k_3 + k_4) * h
        return current_q

class SingleTrack(Robot):
    def __init__(self):
        g = 9.88

        #position state
        s = ca.SX.sym('s') #curvilinear coordinate along the path
        e = ca.SX.sym('e') #lateral distance to the path
        d_psi = ca.SX.sym('d_psi') #difference in heading between vehicle and path
        delta = ca.SX.sym('delta') #front steer angle TODO understand this better, is it part of the state? In the paper it's not presented as part of the state, but it is used in the dynamic equations so I guess it is
        #velocity states
        Ux = ca.SX.sym('Ux')
        Uy = ca.SX.sym('Uy')
        r = ca.SX.sym('r') #yaw rate
        q = ca.vertcat(Ux, Uy, r, s,e,d_psi, delta)
        self.state_labels=['Ux', 'Uy','r','s', 'e','d_psi',  'delta']

        #input
        Fx = ca.SX.sym('Fx') #longitudinal force command
        delta_dot = ca.SX.sym('delta_dot') #steering angle rate
        u = ca.vertcat(Fx, delta_dot)
        self.input_labels=['Fx','delta_dot']

        Parameters = namedtuple('Parameters', ['m', 'a', 'b', 'h_cg', 'Izz'])
        p = Parameters(1000, 2, 2, 1, 2500)
        
        #utils for ODE
        Xf, Xr = self.get_force_distirbution(Fx)
        Fx_f = Fx*Xf
        Fx_r = Fx*Xr
        #TODO in the future should I use Fx_f Fx_r? or Fx=vertcat(), or just Fx? 

        #TODO theta and phi are road grade and bank angle, but for now we assume flat track
        theta = 0 
        phi = 0
        Av2 = 0 #because theta, phi are 0
        Fb = -p.m*g*ca.cos(theta)*ca.sin(phi) 
        Fn = -p.m*g*ca.cos(theta)*ca.sin(phi)
        Crr = 0.014 #https://en.wikipedia.org/wiki/Rolling_resistance
        Frr = Crr*Fn #rolling resistance = coefficient*normal force (not specified in the paper)
        Cd = 0.25 #https://en.wikipedia.org/wiki/Automobile_drag_coefficient#:~:text=The%20average%20modern%20automobile%20achieves,a%20Cd%3D0.35%E2%80%930.45.
        Fd = Frr + Cd*Ux**2 - p.m*g*ca.sin(theta) 
        L = (p.a+p.b)
        #TODO for example here should I use Fx_f for Fz_f and Fx_r for Fz_r? Or just the value Fx
        Fz_f = p.b/L * p.m*(g*ca.cos(theta)*ca.cos(phi) + Av2*Ux**2) - p.h_cg*Fx/L 
        Fz_r = p.a/L*p.m*(g*ca.cos(theta)*ca.cos(phi)+Av2*Ux**2) + p.h_cg*Fx/L
        #The paper gives formula for Fz_f Fz_r, alpha_f alpha_r. Fz is just the vertcat? I'm not sure

        Fz = ca.vertcat(Fz_f, Fz_r) #I think this is the corret way to do it?

        alpha_f = ca.atan((Uy + p.a*r)/Ux) - delta
        alpha_r = ca.atan((Uy - p.b*r)/Ux)
        alpha = ca.vertcat(alpha_f, alpha_r)

        Fy = self.get_lateral_force(alpha, Fz, Fx)
        Fy_f = Fy[0]
        Fy_r = Fy[1]
        k = 0 #assume straight line
        # ODE
        
        Ux_dot = (Fx_f*ca.cos(delta) - Fy_f * ca.sin(delta) - Fd)/p.m + r*Uy
        Uy_dot = (Fy_f * ca.cos(delta) + Fx_f*ca.sin(delta) + Fy_r + Fb)/p.m - r*Ux
        r_dot = (p.a*(Fy_f*ca.cos(delta) + Fx_f*ca.sin(delta)) - p.b*Fy_r) / p.Izz #TODO moment of inertia? maybe from here? http://archive.sciendo.com/MECDC/mecdc.2013.11.issue-1/mecdc-2013-0003/mecdc-2013-0003.pdf
        s_dot = (Ux*ca.cos(d_psi) - Uy*ca.sin(d_psi)) / (1 - k* e) #TODO The path’s curvature κ defines the horizontal path geometry
        e_dot = Ux*ca.sin(d_psi) + Uy*ca.cos(d_psi)
        d_psi_dot = r - k*s_dot
        delta_dot = 0 #added because we need same dimensionality? TODO UNDERSTAND


        qd = ca.vertcat(Ux_dot, Uy_dot, r_dot, s_dot, e_dot, d_psi_dot, delta_dot)

        super().__init__(q,u,qd)

    def get_force_distirbution(self, Fx):
        #for rear only drive:
        df = 0
        dr = 1
        bf = 0.5
        br = 0.5 
        Xf = (df-bf)/2 * ca.tanh(2*(Fx + 0.5)) + (df + bf)/2
        Xr = (br-dr)/2 * ca.tanh(-2*(Fx + 0.5)) + (dr + br)/2 
        return Xf, Xr

    def get_lateral_force(self, alpha, Fz, Fx):
        eps = 0.85 #from the paper

        mu = 0.4 #http://hyperphysics.phy-astr.gsu.edu/hbase/Mechanics/frictire.html
        Fy_max = ((mu*Fz)**2 - (0.99*Fx)**2)**0.5 
        C_alpha = 3.5 #https://www.politesi.polimi.it/bitstream/10589/152539/3/2019_12_Viani.pdf  proposes some real value taken from wherever, we should look at this in more detail

        alpha_mod = ca.atan(3*Fy_max*eps/C_alpha) #alpha mod is 2D because Fymax is 2D because Fz is 2D
        #does this make sense? :
        #for example here I have to use the two different values of alpha?
        alpha_f = alpha[0]
        alpha_r = alpha[1]

        condition = ca.logic_and((ca.fabs(alpha_f) <= alpha_mod[0]) , (ca.fabs(alpha_r) <= alpha_mod[1]))

        # Element-wise operations based on conditions
        result = ca.if_else(
            condition,
            ca.vertcat(
            -C_alpha*ca.tan(alpha_f) + C_alpha**2*ca.fabs(ca.tan(alpha_f))*ca.tan(alpha_f) / (3*Fy_max),
            -C_alpha*ca.tan(alpha_r) + C_alpha**2*ca.fabs(ca.tan(alpha_r))*ca.tan(alpha_r) / (3*Fy_max)
            ),
            ca.vertcat(
            -C_alpha*(1 - 2*eps + eps**2)*ca.tan(alpha_f) - Fy_max*(3*eps**2 - 2*eps**3)*ca.sign(alpha_f),
            -C_alpha*(1 - 2*eps + eps**2)*ca.tan(alpha_r) - Fy_max*(3*eps**2 - 2*eps**3)*ca.sign(alpha_r)
            )
        )
        return result
        

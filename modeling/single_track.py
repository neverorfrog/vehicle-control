import casadi as ca
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from modeling.util import *
from collections import namedtuple

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
        self.length = 0.4
        g = 9.88
        #velocity states
        Ux = ca.SX.sym('Ux') #longitudinal speed
        Uy = ca.SX.sym('Uy') #lateral speed
        r = ca.SX.sym('r') #yaw rate

        #position state
        s = ca.SX.sym('s') #curvilinear coordinate along the path
        e = ca.SX.sym('e') #lateral distance to the path
        d_psi = ca.SX.sym('d_psi') #difference in heading between vehicle and path
        delta = ca.SX.sym('delta') #front steer angle 
        q = ca.vertcat(Ux, Uy, r, s, e, d_psi, delta)
        self.state_labels=['Ux','Uy','r','s','e','d_psi','delta']

        #input
        Fx = ca.SX.sym('Fx') #longitudinal force command
        delta_dot_cmd = ca.SX.sym('delta_dot') #steering angle rate
        u = ca.vertcat(Fx, delta_dot_cmd)
        self.input_labels=['Fx','delta_dot']

        #parameters: mass, distance of CG from front (a) and rear (b) axis, height of CG, yaw moment of inertia
        Parameters = namedtuple('Parameters', ['m', 'a', 'b', 'h_cg', 'Izz'])
        p = Parameters(1000, 2, 2, 1, 2500)
        
        #utils for ODE
        Xf, Xr = self.get_force_distirbution(Fx)

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
        Fy = self.get_lateral_force(alpha_f, alpha_r,Uy, Ux, delta, r ,Fz_f, Fz_r, Fx, Xf)
        
        #path curvature
        k = 0 #assume straight line atm
        # ODE
        #equations 1a to 1f
        Ux_dot = (Fx*Xf(Fx)*ca.cos(delta) - Fy(Uy, Ux, delta, r, Fx)[0] * ca.sin(delta) + Fx*Xr(Fx) - Fd(Ux))/p.m + r*Uy
        Uy_dot = (Fy(Uy, Ux, delta, r, Fx)[0] * ca.cos(delta) + Fx*Xf(Fx)*ca.sin(delta) + Fy(Uy, Ux, delta, r, Fx)[1] + Fb)/p.m - r*Ux
        r_dot = (p.a*(Fy(Uy, Ux, delta, r, Fx)[0]*ca.cos(delta) + Fx*Xf(Fx)*ca.sin(delta)) - p.b*Fy(Uy, Ux, delta, r, Fx)[1]) / p.Izz #TODO moment of inertia? maybe from here? http://archive.sciendo.com/MECDC/mecdc.2013.11.issue-1/mecdc-2013-0003/mecdc-2013-0003.pdf
        s_dot = (Ux*ca.cos(d_psi) - Uy*ca.sin(d_psi)) / (1 - k* e) #TODO The path’s curvature κ defines the horizontal path geometry
        e_dot = Ux*ca.sin(d_psi) + Uy*ca.cos(d_psi)
        d_psi_dot = r - k*s_dot
        delta_dot = delta_dot_cmd #TODO UNDERSTAND


        qd = ca.vertcat(Ux_dot, Uy_dot, r_dot, s_dot, e_dot, d_psi_dot, delta_dot)

        super().__init__(q,u,qd)

    def get_force_distirbution(self, Fx):
        """
        returns two casadi functions, to get the front/rear force distribution given Fx
        equation 6a/6b in the paper
        """
        #for rear only drive:
        df = 0
        dr = 1
        bf = 0.5
        br = 0.5 
        Xf = ca.Function("force_distirbution", [Fx], [(df-bf)/2 * ca.tanh(2*(Fx + 0.5)) + (df + bf)/2])
        Xr = ca.Function("force_distirbution", [Fx], [(br-dr)/2 * ca.tanh(-2*(Fx + 0.5)) + (dr + br)/2 ])
        return Xf, Xr

    def get_lateral_force(self, alpha_f, alpha_r, Uy, Ux, delta, r, Fz_f, Fz_r, Fx, Xf):
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
    
def plot(self, axis: Axes, q = None):
        x,y,v,psi,delta,_,_,_,_ = self.state.values if q is None else q
        r = self.length / 2
        
        # Draw the bicycle as a rectangle
        width = self.length
        height = self.length
        angle = wrap(psi-np.pi/2)
        rectangle = plt.Rectangle((x-np.cos(angle)*width/2-np.cos(psi)*2*width/3, y-np.sin(angle)*height/2-np.sin(psi)*2*height/3),
                                  width,height,edgecolor='black',alpha=0.7, angle=np.rad2deg(angle), rotation_point='xy')
        axis.add_patch(rectangle)
        
        # Plot directional tick
        line_length = 1.5 * r
        line_end_x = x + line_length * np.cos(psi)
        line_end_y = y + line_length * np.sin(psi)
        axis.plot([x, line_end_x], [y, line_end_y], color='r', lw=3)
        
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
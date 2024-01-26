# inspired by https://github.com/giulioturrisi/Differential-Drive-Robot/blob/main/python_scripts/controllers/casadi_nmpc.py
from matplotlib.pyplot import plot
from modeling.kinematic_car import KinematicCar
import casadi as ca
from modeling.track import Track
import numpy as np
from casadi import cos, sin, tan
from controllers.controller import Controller
from modeling.util import integrate

class RacingMPC(Controller):
    def __init__(self, horizon, dt, car: KinematicCar):
        self.dt = dt
        self.car = car
        self.horizon = horizon

        # -------------------- Optimizer Initialization ----------------------------------
        self.opti = ca.Opti()
        ns, ni, transition = self._init_ode()
        p_opts = dict(print_time=False, verbose=False) 
        s_opts = dict(print_level=1)
        self.opti.solver("ipopt", p_opts, s_opts)
        
        # -------------------- Decision Variables with Initialization ---------------------
        self.state = self.opti.variable(ns, horizon+1) # predicted state trajectory var
        self.action = self.opti.variable(ni, horizon)   # predicted control trajectory var
        self.state_prediction = np.ones((ns, horizon+1)) # actual predicted state trajectory
        self.action_prediction = np.ones((ni, horizon))   # actual predicted control trajectory
        
        # --------------------- Helper Variables ------------------------------------------
        self.s0 = self.opti.parameter(ns) # initial state
        self.opti.subject_to(self.state[:,0] == self.s0) # constraint on initial state
        self.kappa = self.opti.parameter(horizon) # local curvature
        # self.opti.set_value(self.kappa, np.zeros(horizon))
        
        # -------------------- Model Constraints and Cost function ------------------------
        for n in range(horizon):
            state = self.state[:,n]
            self.opti.subject_to(state[2] >= 1) # TODO without this, things break
            
        cost = 0
        for n in range(horizon):
            
            # extracting state
            state = self.state[:,n]
            input = self.action[:,n]
            state_next = self.state[:,n+1]
            v = state[2] # TODO hardcodato
            ey = state[5] # TODO hardcodato
            epsi = state[6] # TODO hardcodato
            
            # add stage cost to objective
            cost += 4*ca.sumsqr(ey - 0.3) # right bound
            cost += 4*ca.sumsqr(ey + 0.3) # left bound
            cost += ca.sumsqr(epsi - ca.pi/3) #
            cost += ca.sumsqr(epsi + ca.pi/3) # 
            cost += 5*ca.sumsqr(epsi) #
            cost += ca.sumsqr(ey) # 

            # going on for dt and snapshot of how much the car moved
            # self.ds = self.dt * ((v * np.cos(epsi)) / (1 - ey * self.kappa[n]))
            self.ds = 0.01 # TODO bigger step breaks things
            
            # add continuity contraint on spatial dynamics
            self.opti.subject_to(state_next == transition(state,input,self.kappa[n],self.ds))
            
        cost += 0.1*state[-1,-1] # final cost (minimize time) # TODO bigger weight breaks things
        cost += 100*ca.sumsqr(state[5]) # final cost (minimize terminal error)
        cost += 100*ca.sumsqr(state[6]) # final cost (minimize terminal error)
        self.opti.minimize(cost)
            
        # -------------------- Model Constraints ------------------------------------------
        self.a_max = 1 # TODO higher value breaks things
        self.a_min = -0.1 # TODO higher value breaks things
        self.w_max = 2 # TODO higher value breaks things
        self.w_min = -1.5 # TODO lower value breaks things
        for n in range(horizon): # loop over control intervals
            self.opti.subject_to(self.action[0,n] <= self.a_max)
            self.opti.subject_to(self.action[0,n] >= self.a_min)
            self.opti.subject_to(self.action[1,n] <= self.w_max)
            self.opti.subject_to(self.action[1,n] >= self.w_min)
            
    def _init_ode(self):
        '''Differential equations describing the model inside the prediction horizon'''
        ns = 8 # number of states -> (v,psi,delta,ey,epsi,t)
        ni = 2 # number of inputs -> (a,w)
        
        # input variables
        a = ca.SX.sym('a') # driving acceleration
        w = ca.SX.sym('w') # steering angle rate
        input = ca.vertcat(a,w)
        
        # state and auxiliary variables
        x,y,v,psi,delta,s,ey,epsi,t = self.car.state.variables
        kappa = ca.SX.sym('kappa')
        ds = ca.SX.sym('ds')
        state = ca.vertcat(x,y,v,psi,delta,ey,epsi,t)
        
        # ODE
        x_prime = ((1 - ey*kappa) * cos(psi)) / (cos(epsi))
        y_prime = ((1 - ey*kappa) * sin(psi)) / (cos(epsi))
        psi_prime = ((1 - ey*kappa) * tan(delta)) / (self.car.length * cos(epsi))
        v_prime = (1 - ey * kappa) / (v * np.cos(epsi)) * a
        delta_prime = (1 - ey * kappa) / (v * np.cos(epsi)) * w
        ey_prime = (1 - ey * kappa) * ca.tan(epsi)
        epsi_prime = ((tan(delta)) / self.car.length) * ((1 - ey * kappa)/(np.cos(epsi))) - kappa
        t_prime = (1 - ey * kappa) / (v * np.cos(epsi))
        state_prime = ca.vertcat(x_prime, y_prime, v_prime, psi_prime, delta_prime, ey_prime, epsi_prime, t_prime)
        ode = ca.Function('ode', [state, input], [state_prime],{'allow_free':True})
        
        # wrapping up
        integrator = integrate(state, input, ode, h=ds) # TODO ds
        transition = ca.Function('transition', [state,input,kappa,ds], [integrator])
        
        return ns, ni, transition
        
    def command(self, s_k, kappa):
        # every new horizon, the current state (and the last prediction) is the initial prediction
        self.start(s_k, kappa)
        sol = self.opti.solve()
        # print(self.opti.debug.value(self.ds))
        self.action_prediction = sol.value(self.action)
        self.state_prediction = sol.value(self.state)
        return np.array([self.action_prediction[0][0], self.action_prediction[1][0]]), self.state_prediction
    
    def start(self, s_k, kappa):
        '''
        s_k is a state in its entirety. I need to extract relevant variables
        '''
        state = np.array([s_k.x, s_k.y, s_k.v, s_k.psi, s_k.delta, s_k.ey, s_k.epsi, s_k.t])
        self.opti.set_value(self.kappa, kappa)
        self.opti.set_value(self.s0, state)
        self.opti.set_value(self.kappa, self.car.current_waypoint.kappa)
        self.opti.set_initial(self.action, self.action_prediction)
        self.opti.set_initial(self.state, self.state_prediction)
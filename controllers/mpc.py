# inspired by https://github.com/giulioturrisi/Differential-Drive-Robot/blob/main/python_scripts/controllers/casadi_nmpc.py
from matplotlib.pyplot import plot
from modeling.racing_car import KinematicCar
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
        s_opts = dict(print_level=0)
        self.opti.solver("ipopt", p_opts, s_opts)
        
        # -------------------- Decision Variables with Initialization ---------------------
        self.state = self.opti.variable(ns, horizon+1) # predicted state trajectory var
        self.action = self.opti.variable(ni, horizon)   # predicted control trajectory var
        self.state_prediction = np.zeros((ns, horizon+1)) # actual predicted state trajectory
        self.action_prediction = np.zeros((ni, horizon))   # actual predicted control trajectory
        
        # --------------------- Helper Variables ------------------------------------------
        self.s0 = self.opti.parameter(ns) # initial state
        self.opti.subject_to(self.state[:,0] == self.s0) # constraint on initial state
        self.kappa = self.opti.parameter(1) # local curvature
        self.ds = self.opti.variable(1)
        
        # -------------------- Model Constraints and Cost function ------------------------
        for n in range(horizon):
            state = self.state[:,n]
            self.opti.subject_to(state[0] >= 1)
            
        cost = 0
        for n in range(horizon):
            state = self.state[:,n]
            input = self.action[:,n]
            state_next = self.state[:,n+1]
            
            # add stage cost to objective
            t = state[-1]
            cost += t # stage cost (minimize time)

            # going on for dt and snapshot of how much the car moved
            v = state[0]
            ey = state[3]
            epsi = state[4]
            self.ds = self.dt * (v * np.cos(epsi)) / (1 - ey * self.kappa)
            
            # add continuity contraint on spatial dynamics
            # self.opti.subject_to(state_next == transition(state,input,self.kappa,self.ds))
            
        self.opti.minimize(cost)
            
        # -------------------- Input Constraints ------------------------------------------
        # self.v_max = 5.0
        # self.v_min = -5.0
        # self.w_max = 5.0
        # self.w_min = -5.0
        # for n in range(horizon): # loop over control intervals
        #     self.opti.subject_to(self.action[0,n] <= self.v_max)
        #     self.opti.subject_to(self.action[0,n] >= self.v_min)
        #     self.opti.subject_to(self.action[1,n] <= self.w_max)
        #     self.opti.subject_to(self.action[1,n] >= self.w_min)
            
    def _init_ode(self):
        '''Differential equations describing the model inside the prediction horizon'''
        ns = 6 # number of states -> (v,psi,delta,ey,epsi,t)
        ni = 2 # number of inputs -> (a,w)
        
        # input variables
        a = ca.SX.sym('a') # driving acceleration
        w = ca.SX.sym('w') # steering angle rate
        input = ca.vertcat(a,w)
        
        # state and auxiliary variables
        x,y,v,psi,delta,s,ey,epsi,t = self.car.state.variables
        kappa = ca.SX.sym('kappa')
        ds = ca.SX.sym('ds')
        state = ca.vertcat(v,psi,delta,ey,epsi,t)
        # state = ca.vertcat(v)
        
        # ODE
        v_prime = (1 - ey * kappa) / (v * np.cos(epsi)) * a
        psi_prime = (1 - ey * kappa) / (v * np.cos(epsi) * self.car.length * tan(delta))
        delta_prime = ((1 - ey * kappa) / (v * np.cos(epsi))) * w
        ey_prime = (1 - ey * kappa) * ca.tan(epsi)
        epsi_prime = (self.car.length * ca.tan(delta)) * ((1 - ey * kappa) / (np.cos(epsi))) - kappa
        t_prime = (1 - ey * kappa) / (v * np.cos(epsi))
        state_prime = ca.vertcat(v_prime, psi_prime, delta_prime, ey_prime, epsi_prime, t_prime)
        # state_prime = ca.vertcat(v_prime)
        ode = ca.Function('ode', [state, input], [state_prime],{'allow_free':True})
        
        # wrapping up
        integrator = integrate(state, input, ode, h=ds) # TODO ds
        transition = ca.Function('transition', [state,input,kappa,ds], [integrator])
        
        return ns, ni, transition
        
    def command(self, s_k):
        # every new horizon, the current state (and the last prediction) is the initial prediction
        [print(s_k)]
        s_k = self.convert_state(s_k)
        self.opti.set_value(self.s0, s_k) 
        print(self.opti.debug.value(self.s0,self.opti.initial()))
        self.opti.set_value(self.kappa, self.car.current_waypoint.kappa)
        self.opti.set_initial(self.action, self.action_prediction)
        self.opti.set_initial(self.state, self.state_prediction)
        sol = self.opti.solve()
        print(self.opti.debug.value(self.action))
        self.action_prediction = sol.value(self.action)
        self.state_prediction = sol.value(self.state)
        return np.array([self.action_prediction[0][0], self.action_prediction[1][0]])  
    
    def convert_state(self, s_k):
        '''
        s_k is a state in its entirety. I need to extract relevant variables
        '''
        # return np.array([s_k.v])
        return np.array([s_k.v, s_k.psi, s_k.delta, s_k.ey, s_k.epsi, s_k.t])
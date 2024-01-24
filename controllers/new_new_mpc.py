# inspired by https://github.com/giulioturrisi/Differential-Drive-Robot/blob/main/python_scripts/controllers/casadi_nmpc.py
import sys
sys.path.append("..")

from modeling.bicycle import Bicycle
import casadi as ca
from modeling.track import Track
import numpy as np
from controllers.controller import Controller
from casadi import cos, sin, tan

class RacingMPC(Controller):
    def __init__(self, horizon, dt, car: Bicycle):
        self.dt = dt
        self.car = car
        self.horizon = horizon
        ns = 5 # number of states
        ni = 4 # number of inputs
        # -------------------- Optimizer Initialization ----------------------------------
        self.opti = ca.Opti()
        p_opts = dict(print_time=False, verbose=False) 
        s_opts = dict(print_level=0)
        self.opti.solver("ipopt", p_opts, s_opts)
        
        # -------------------- Decision Variables with Initialization ---------------------
        self.S = self.opti.variable(ns, horizon+1) # predicted state trajectory var
        self.I = self.opti.variable(ni, horizon)   # predicted control trajectory var
        self.S_pred = np.zeros((ns, horizon+1)) # actual predicted state trajectory
        self.I_pred = np.zeros((ni, horizon))   # actual predicted control trajectory
        self.s0 = self.opti.parameter(ns) # initial state
        self.opti.subject_to(self.S[:,0] == self.s0) # constraint on initial state
        
        # -------------------- Model Constraints and Cost function ------------------------
        cost = 0
        for n in range(horizon):
            state = self.S[:,n]
            input = self.I[:,n]
            state_next = self.S[:,n+1]
            
            # add stage cost to objective
            cost += ca.sumsqr(state[:2]) # stage cost (make x and y and psi zero)

            # add continuity contraint
            self.opti.subject_to(state_next == car.t_transition(state,input))
            
        self.opti.minimize(cost)
            
        # -------------------- Input Constraints ------------------------------------------
        self.v_max = 2.0
        self.v_min = -2.0
        self.w_max = 2.0
        self.w_min = -2.0
        for n in range(horizon): # loop over control intervals
            self.opti.subject_to(self.I[0,n] <= self.v_max)
            self.opti.subject_to(self.I[0,n] >= self.v_min)
            self.opti.subject_to(self.I[1,n] <= self.w_max)
            self.opti.subject_to(self.I[1,n] >= self.w_min)
        
    def command(self, s_k):
        # every new horizon, the current state (and the last prediction) is the initial prediction
        self.opti.set_value(self.s0, s_k) 
        self.opti.set_initial(self.I, self.I_pred)
        self.opti.set_initial(self.S, self.S_pred)
        sol = self.opti.solve()
        self.I_pred = sol.value(self.I)
        self.S_pred = sol.value(self.S)
        # print(self.opti.debug.value)
        delta = self.S_pred[3,-1]
        return np.array([self.I_pred[0][0], self.I_pred[1][0], delta, self.car.current_waypoint.kappa])
    
if __name__ =="__main__":
    # Create reference path
    wp = np.array([[-2,0],[2,0],[2,2],[-2,2],[-2,0],[0,0]])
    track = Track(wp_x=wp[:,0], wp_y=wp[:,1], resolution=0.05,smoothing=15, width=0.15)
    
    # Bicycle model
    car = Bicycle(track, length=0.5, dt=0.05)
    controller = RacingMPC(4,0.1,car)     
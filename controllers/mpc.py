# inspired by https://github.com/giulioturrisi/Differential-Drive-Robot/blob/main/python_scripts/controllers/casadi_nmpc.py
import sys
sys.path.append("..")

import casadi as ca
import numpy as np
from modeling.car import Car
from controllers.controller import Controller
from casadi import cos, sin, tan

class RacingMPC(Controller):
    def __init__(self, horizon, dt, car: Car):
        self.dt = dt
        self.car = car
        # -------------------- Optimizer Initialization ----------------------------------
        self.N = horizon
        self.opti = ca.Opti()
        p_opts = dict(print_time=False, verbose=False) 
        s_opts = dict(print_level=0)
        self.opti.solver("ipopt", p_opts, s_opts)
        
        # -------------------- Decision Variables with Initialization ---------------------
        self.X = self.opti.variable(car.q_len, self.N+1) # predicte state trajectory var
        self.U = self.opti.variable(car.u_len, self.N)   # predicted control trajectory var
        self.X_pred = np.zeros((car.q_len, self.N+1)) # actual predicte state trajectory
        self.U_pred = np.zeros((car.u_len, self.N))   # actual predicted control trajectory
        self.x0 = self.opti.parameter(car.q_len)
        self.opti.subject_to(self.X[:,0] == self.x0) # constraint on initial state
        
        # -------------------- Model Constraints (ODE) ------------------------------------
        # Differential equations describing the model during planning phase
        self.k = self.opti.parameter(1)
        v_prime       = ((1 - car.ey*self.k) / (car.v * cos(car.epsi))) * car.a
        psi_prime     = (tan(car.delta)*(1 - car.ey*self.k) / (car.l * cos(car.epsi)))
        t_prime       = ((car.v * cos(car.epsi)) / (1 - car.ey*self.k))
        ey_prime      = (1 - car.ey*self.k) * tan(car.epsi)
        epsi_prime    = ((1 - car.ey*self.k) / (cos(car.epsi))) * car.psi - self.k
        delta_prime   = ((1 - car.ey*self.k) / (car.v * cos(car.epsi))) * car.w
        s_prime       = 1
        self.qd     = ca.vertcat(v_prime,psi_prime,t_prime,ey_prime,epsi_prime,delta_prime,s_prime)
        self.ode = ca.Function('ode', [car.q, car.u], [self.qd], {'allow_free': True})
        # Direct transcription in space (and not in time)
        # x(s + h) = x(s) + h * f(x(s),u(s)) (actually i'll use RK4)
        # h is how much i travel along the road descriptor at current velocity in 30ms
        for k in range(self.N): # loop over control intervals
            v = self.X[0,k]
            epsi = self.X[4,k]
            ey = self.X[3,k]
            h = 0.03 * (v * cos(epsi)) / (1 - ey*self.k) # compute s shift by euler integration with fixed velocity
            transition = ca.Function('transition', [self.car.q,self.car.u], [self.car.integrate(h)], {'allow_free': True})
            next_x = transition(self.X[:,k],self.U[:,k])
            self.opti.subject_to(self.X[:,k+1] == next_x)
            
        # -------------------- Cost Function -----------------------------------------------
        cost = self.X[2,-1] # time to arrive at last planning step
        
        self.ws = 10
        cost += self.ws * ca.sumsqr(self.X[3,:])
        
        # self.wu = 1
        # self.ref = self.opti.parameter(2,self.N+1) # TODO hardcodato a due reference (posizione)
        # for k in range(1, self.N):
        #     ref_k = self.ref[:,k-1]
        #     cost =  self.wpos*ca.sumsqr(self.X[0,:]-ref_k[0]) + self.wpos*ca.sumsqr(self.X[1,:]-ref_k[1])
        # cost += self.wu*ca.sumsqr(self.U[:,0]) + self.wu*ca.sumsqr(self.U[:,1])
        self.opti.minimize(cost)
            
        # -------------------- Input Constraints ------------------------------------------
        self.a_max = 10.0
        self.w_max = 10.0
        self.u_k = np.array([0., 0.])
        for k in range(self.N): # loop over control intervals
            #linear acceleration
            self.opti.subject_to(self.U[0,k] <= self.a_max)
            self.opti.subject_to(self.U[0,k] >= -self.a_max)
            #angular velocity
            self.opti.subject_to(self.U[1,k] <= self.w_max)
            self.opti.subject_to(self.U[1,k] >= -self.w_max)
        
        
    def command(self, q_k, curvature):
        # q_k -> {'v', 'psi', 't', 'ey', 'epsi', 'delta', 's', 'k'}
        # every new horizon, the current state (and the last prediction) is the initial prediction
        self.opti.set_value(self.x0, q_k) 
        self.opti.set_value(self.k, curvature)
        # self.opti.set_initial(self.U, self.U_pred)
        # self.opti.set_initial(self.X, self.X_pred)
        
        # obtaining the solution by solving the NLP
        # sol = self.opti.solve()
        # self.U_pred = sol.value(self.U)
        # self.X_pred = sol.value(self.X)
        # print(self.opti.debug.value)
        # return np.array([self.U_pred[0][0],self.U_pred[1][0]])
        return np.array([0.04,0.5])        
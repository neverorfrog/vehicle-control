# inspired by https://github.com/giulioturrisi/Differential-Drive-Robot/blob/main/python_scripts/controllers/casadi_nmpc.py
import sys

from modeling.unicycle import Unicycle
sys.path.append("..")

import casadi as ca
import numpy as np
from modeling.car import Car
from controllers.controller import Controller
from casadi import cos, sin, tan

class RacingMPC(Controller):
    def __init__(self, horizon, dt, car: Unicycle):
        self.dt = dt
        self.car = car
        self.v = 10
        self.w = 10
        # -------------------- Optimizer Initialization ----------------------------------
        self.N = horizon
        self.opti = ca.Opti()
        p_opts = dict(print_time=False, verbose=False) 
        s_opts = dict(print_level=0)
        self.opti.solver("ipopt", p_opts, s_opts)
        
        # -------------------- Decision Variables with Initialization ---------------------
        q_len = car.q_len
        u_len = car.u_len
        self.X = self.opti.variable(q_len, self.N+1) # predicte state trajectory var
        self.U = self.opti.variable(u_len, self.N)   # predicted control trajectory var
        self.X_pred = np.zeros((q_len, self.N+1)) # actual predicte state trajectory
        self.U_pred = np.zeros((u_len, self.N))   # actual predicted control trajectory
        self.x0 = self.opti.parameter(q_len)
        self.opti.subject_to(self.X[:,0] == self.x0) # constraint on initial state
        
        # -------------------- Model Constraints (ODE) ------------------------------------
        self.propagate_model(car)
            
        # -------------------- Cost Function -----------------------------------------------
        # cost = self.X[-1,-1] # time to arrive at last planning step
        ws = 10
        # cost = ws * ca.sumsqr(self.X[5,:])
        wu = 10
        cost = wu*ca.sumsqr(self.U[:,0]) + wu*ca.sumsqr(self.U[:,1])
        self.opti.minimize(cost)
        
        # -------------------- Termination Constraints ------------------------------------------
        x_ter = np.array((0.5, 0, 0, 0))
        self.opti.subject_to(self.X[0,self.N] == x_ter[0])
            
        # -------------------- Input Constraints ------------------------------------------
        # self.v_max = 10.0
        # self.w_max = 10.0
        # self.u_k = np.array([0., 0.])
        # for k in range(self.N): # loop over control intervals
        #     #linear acceleration
        #     self.opti.subject_to(self.U[0,k] <= self.v_max)
        #     self.opti.subject_to(self.U[0,k] >= -self.v_max)
        #     #angular velocity
        #     self.opti.subject_to(self.U[1,k] <= self.w_max)
        #     self.opti.subject_to(self.U[1,k] >= -self.w_max)
        
    def integrate(self, q, u, h):
        '''
        RK4 integrator
        h: integration interval
        '''
        qd_1 = self.ode(q, u)
        qd_2 = self.ode(q + (h/2)*qd_1, u)
        qd_3 = self.ode(q + (h/2)*qd_2, u)
        qd_4 = self.ode(q + h*qd_3, u)
        q += (1/6) * (qd_1 + 2 * qd_2 + 2 * qd_3 + qd_4) * h
        return q
    
    def propagate_model(self, car: Car):
        # TODO can v and w be taken in this way ?
        # Differential equations describing the model during planning phase
        k = 0 # self.opti.parameter(1) # TODO this way the curvature is the same for the whole planning horizon
        epsi = car.epsi
        ey = car.ey
        v = self.v
        w = self.w
        self.x_prime      = ((1 - k*ey) * cos(car.psi)) / cos(epsi)
        self.y_prime      = ((1 - k*ey) * sin(car.psi)) / cos(epsi)
        self.psi_prime    = ((1 - k*ey) / (v * cos(epsi))) * w
        self.s_prime      = 1
        self.ey_prime     = tan(epsi) * (1 - k*ey)
        self.epsi_prime   = ((1 - k*ey) / (v * cos(epsi))) * w - k
        self.t_prime      = ((1 - k*ey) / (v * cos(epsi)))
        self.qd = ca.vertcat(self.x_prime, self.y_prime, self.psi_prime, self.s_prime, self.ey_prime, self.epsi_prime, self.t_prime)
        self.ode = ca.Function('ode', [car.q, car.u], [self.qd], {'allow_free': True})
        
        # Direct transcription in space (and not in time)
        # x(s + h) = x(s) + h * f(x(s),u(s)) (actually i'll use RK4)
        # h is how much i travel along the road descriptor at current velocity in 30ms
        for k in range(self.N): # loop over control intervals
        #     v = self.X[0,k]
        #     epsi = self.X[4,k]
        #     ey = self.X[3,k]
        #     h = 0.03 * (v * cos(epsi)) / (1 - ey*self.k) # compute s shift by euler integration with fixed velocity
            transition = ca.Function('transition', [car.q,car.u], [self.integrate(car.q,car.u,self.dt)], {'allow_free':True})
            next_x = transition(self.X[:,k],self.U[:,k])
            next_x = self.X[:,k] + self.dt * self.ode(self.X[:,k], self.U[:,k])
            self.opti.subject_to(self.X[:,k+1] == next_x)
        
        
    def command(self, q_k, curvature):
        # q_k -> {'v', 'psi', 't', 'ey', 'epsi', 'delta', 's', 'k'}
        # every new horizon, the current state (and the last prediction) is the initial prediction
        self.opti.set_value(self.x0, q_k) 
        # self.opti.set_value(self.k, curvature)
        self.opti.set_initial(self.U, self.U_pred)
        self.opti.set_initial(self.X, self.X_pred)
        
        # obtaining the solution by solving the NLP
        sol = self.opti.solve()
        self.U_pred = sol.value(self.U)
        self.X_pred = sol.value(self.X)
        print(self.opti.debug.value)
        self.v = self.U_pred[0][0]
        self.w = self.U_pred[1][0]
        return np.array([self.v, self.w])
# taken from https://github.com/giulioturrisi/Differential-Drive-Robot/blob/main/python_scripts/controllers/casadi_nmpc.py
import sys
import time

from modeling.trajectory import Trajectory
sys.path.append("..")

from modeling.robot import *
import casadi as ca
import numpy as np
from controllers.controller import Controller
from numpy import sin,cos,tan

class MPC(Controller):
    def __init__(self, horizon, dt, model: Robot):
        # -------------------- Optimizer Initialization ----------------------------------
        self.N = horizon
        self.opti = ca.Opti()
        p_opts = dict(print_time=False, verbose=False) 
        s_opts = dict(print_level=0)
        self.opti.solver("ipopt", p_opts, s_opts)
        
        
        # -------------------- Decision Variables with Initialization ---------------------
        self.X = self.opti.variable(model.q_len, self.N+1) # predicte state trajectory var
        self.U = self.opti.variable(model.u_len, self.N)   # predicted control trajectory var
        self.X_pred = np.zeros((model.q_len, self.N+1)) # actual predicte state trajectory
        self.U_pred = np.zeros((model.u_len, self.N))   # actual predicted control trajectory
        self.x0 = self.opti.parameter(model.q_len)
        self.opti.subject_to(self.X[:,0] == self.x0) # constraint on initial state
        self.x = self.X[0,:]
        self.y = self.X[1,:]
        self.theta = self.X[2,:]
        
        
        # -------------------- Model Constraints (ODE) ------------------------------------
        self.dt = dt
        self.model_step = ca.Function('model_step', [model.q,model.u], [model.RK4(dt)])
        for k in range(self.N): # loop over control intervals
            next_x = self.model_step(self.X[:,k],self.U[:,k])
            self.opti.subject_to(self.X[:,k+1] == next_x)
            
            
        # -------------------- Input Constraints ------------------------------------------
        self.v_max = 10.0
        self.w_max = 10.0
        self.u_k = np.array([0., 0.])
        for k in range(self.N): # loop over control intervals
            #linear velocity
            self.opti.subject_to(self.U[0,k] <= self.v_max)
            self.opti.subject_to(self.U[0,k] >= -self.v_max)
            #angular velocity
            self.opti.subject_to(self.U[1,k] <= self.w_max)
            self.opti.subject_to(self.U[1,k] >= -self.w_max)
            
            
        # -------------------- Cost Function -----------------------------------------------
        cost = 0
        self.wpos = 100
        self.wu = 1
        self.ref = self.opti.parameter(2,self.N+1) # TODO hardcodato a due reference (posizione)
        for k in range(1, self.N):
            ref_k = self.ref[:,k-1]
            cost =  self.wpos*ca.sumsqr(self.X[0,:]-ref_k[0]) + self.wpos*ca.sumsqr(self.X[1,:]-ref_k[1])
        # cost += self.wu*ca.sumsqr(self.U[:,0]) + self.wu*ca.sumsqr(self.U[:,1])
        self.opti.minimize(cost)
        
        
        
    def command(self, q_k, qd_k, t_k, reference: Trajectory):
        
        # every new horizon, the current state is the initial prediction
        self.opti.set_value(self.x0, q_k) 
        self.opti.set_initial(self.U, self.U_pred)
        self.opti.set_initial(self.X, self.X_pred)
        
        # obtaining reference TODO solution that does not need time
        ref = np.zeros([2,self.N+1])
        for j in range(self.N):
            t_j = t_k + self.dt 
            ref_j = reference.update(t_j)
            ref[:,j] = ref_j['p']
        
        self.opti.set_value(self.ref, ref)
        sol = self.opti.solve()
        self.U_pred = sol.value(self.U)
        self.X_pred = sol.value(self.X)
        return np.array([self.U_pred[0][0],self.U_pred[1][0]])
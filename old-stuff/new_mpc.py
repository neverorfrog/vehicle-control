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
        ns = 5
        na = 4
        # -------------------- Decision Variables with Initialization ---------------------
        # optimization variables
        s_traj = ca.SX.sym('s_traj', ns*(horizon+1), 1)
        a_traj = ca.SX.sym('a_traj', na*horizon, 1)
        # split trajectories to obtain x and u for each stage
        s = ca.vertsplit(s_traj, np.arange(0, ns*(horizon+1)+1, ns))
        a = ca.vertsplit(a_traj, np.arange(0, na*horizon+1, na))
        # initial state
        s0_bar = ca.SX.sym('s0_bar', ns, 1)
        # initial constraint
        constraints = [s[0] - s0_bar] # this difference will be bounded up and down
        
        # -------------------- Model Constraints and Cost function ------------------------
        cost = 0
        for n in range(horizon):
            state = s[n]
            action = a[n]
            state_next = s[n+1]
            
            # add stage cost to objective
            cost += ca.sumsqr(state[:3]) # stage cost

            # add continuity contraint
            constraints += [state_next - car.t_transition(state, action)]
            
        # continuity constraints and bounds on u
        constraints = ca.vertcat(*constraints, a_traj)
        
        # set upper and lower bounds 
        # zeros for continuity constraints, a_max/a_min for control bounds
        self.ubg = ca.vertcat(np.zeros((ns*(horizon+1))), np.ones((horizon*na)))
        self.lbg = ca.vertcat(np.zeros((ns*(horizon+1))), np.ones((horizon*na)))
        
        # initialize current solution guess
        self.s_current = np.zeros((ns*(horizon+1), 1))
        self.a_current = np.zeros((na*horizon, 1))

        self.ocp = {'f': cost, 'x': ca.vertcat(s_traj, a_traj), 'g': constraints, 'p':s0_bar}
        self.solver = ca.nlpsol('solver', 'ipopt', self.ocp)
        
        
    def command(self,s0):
        # solve the NLP
        sol = self.solver(x0=ca.vertcat(self.s_current, self.a_current), lbg=self.lbg, ubg=self.ubg, p=s0)
        
        w_opt = sol['x'].full()
        
        ns = 5
        na = 4

        s_opt = w_opt[:(self.horizon+1)*ns]
        a_opt = w_opt[(self.horizon+1)*ns:]

        cost = sol['f'].full()

        self.s_current = s_opt
        self.a_current = a_opt

        a0_opt = a_opt[:na].squeeze()
        return a0_opt
    
if __name__ =="__main__":
    # Create reference path
    wp = np.array([[-2,0],[2,0],[2,2],[-2,2],[-2,0],[0,0]])
    track = Track(wp_x=wp[:,0], wp_y=wp[:,1], resolution=0.05,smoothing=15, width=0.15)
    
    # Bicycle model
    car = Bicycle(track, length=0.5, dt=0.05)
    controller = RacingMPC(40,0.1,car)     
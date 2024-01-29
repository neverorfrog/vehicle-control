from model.kinematic_car import KinematicCar
import casadi as ca
import numpy as np
from casadi import cos, sin, tan
from controller.controller import Controller
from model.state import KinematicCarInput
from utils.utils import integrate

class KinematicMPC(Controller):
    def __init__(self, car: KinematicCar, config):
        self.dt = config['mpc_dt']
        self.car = car
        self.N = config['horizon']
        self.config = config
        self.ns = len(self.car.state) # number of state variables
        self.na = len(self.car.input) # number of action variables
        self.opti = self._init_opti()
        
    def command(self, state, curvature):
        # every new horizon, the current state (and the last prediction) is the initial 
        print("") # to separate between prints
        self._init_parameters(state, curvature)
        sol = self.opti.solve()
        self.action_prediction = sol.value(self.action)
        self.state_prediction = sol.value(self.state)
        curvature = sol.value(self.curvature)
        s = sol.value(self.s)
        print(f"EY PREDICTION: {self.state_prediction[self.car.state.index('ey'),:]}")
        print(f"EPSI PREDICTION: {self.state_prediction[self.car.state.index('epsi'),:]}")
        print(f"DELTA PREDICTION: {self.state_prediction[self.car.state.index('delta'),:]}")
        print(f"OMEGA PREDICTION: {self.action_prediction[1,:]}")
        print(f"CURVATURE PREDICTION: {curvature}")
        print(f"S PREDICTION: {s}")
        next_input = KinematicCarInput(a=self.action_prediction[0][0], w=self.action_prediction[1][0])
        return next_input, self.state_prediction
    
    def _init_parameters(self, state, curvature):
        '''
        s_k is a state in its entirety. I need to extract relevant variables
        '''
        self.opti.set_initial(self.curvature[0], curvature)
        self.opti.set_value(self.state0, state)
        self.opti.set_initial(self.action, self.action_prediction)
        self.opti.set_initial(self.state, self.state_prediction)
        
    def _init_opti(self):
        # -------------------- Optimizer Initialization ----------------------------------
        opti = ca.Opti()
        p_opts = {"ipopt.print_level": 0, "expand":False}
        s_opts = {"max_iter": 100}
        opti.solver("ipopt", p_opts, s_opts)
        
        # -------------------- Decision Variables with Initialization ---------------------
        self.state = opti.variable(self.ns, self.N+1) # predicted state trajectory var
        self.action = opti.variable(self.na, self.N)   # predicted control trajectory var
        self.state_prediction = np.ones((self.ns, self.N+1)) # actual predicted state trajectory
        self.action_prediction = np.ones((self.na, self.N))   # actual predicted control trajectory
        
        # --------------------- Helper Variables ------------------------------------------
        self.state0 = opti.parameter(self.ns) # initial state
        opti.subject_to(self.state[:,0] == self.state0) # constraint on initial state
        self.ds = opti.variable(self.N) #ds trajectory during the planning horizon
        self.s = opti.variable(self.N+1)
        opti.subject_to(self.s[0] == self.state0[self.car.state.index('s')])
        self.curvature = opti.variable(self.N+1)
        
        # -------------------- Model Constraints ------------------------------------------
        state_constraints = self.config['state_constraints']
        for n in range(self.N):
            state = self.state[:,n]
            state_next = self.state[:,n+1]
            input = self.action[:,n]
            
            # state constraints
            opti.subject_to(state[self.car.state.index('v')] >= state_constraints['v_min']) # TODO without this, things break
            opti.subject_to(state[self.car.state.index('delta')] <= state_constraints['delta_max'])
            opti.subject_to(state[self.car.state.index('delta')] >= state_constraints['delta_min'])
            
            # continuity contraint on spatial dynamics
            v = state[self.car.state.index('v')]
            ey = state[self.car.state.index('ey')]
            epsi = state[self.car.state.index('epsi')]
            opti.subject_to(self.ds[n] == self.dt * ((v * np.cos(epsi)) / (1 - ey * self.curvature[n]))) # going on for dt and snapshot of how much the car moved
            opti.subject_to(self.s[n+1] == self.s[n] + self.ds[n])
            opti.subject_to(self.curvature[n] == self.car.curvature(self.s[n]))
            opti.subject_to(state_next == self.car.spatial_transition(state,input,self.curvature[n],self.ds[n]))
            
        # ------------------- Cost Function --------------------------------------
            
        cost_weights = self.config['cost_weights'] 
        cost = 0
        for n in range(self.N):
            state = self.state[:,n]
            input = self.action[:,n]
            ey = state[self.car.state.index('ey')]
            
            # violation of road bounds
            cost += ca.if_else(ey < state_constraints['ey_min'],
                       cost_weights['boundary']*self.ds[n]*(ey - state_constraints['ey_min'])**2, 0)
            
            cost += ca.if_else(ey > state_constraints['ey_max'],
                       cost_weights['boundary']*self.ds[n]*(ey - state_constraints['ey_max'])**2, 0)
                
            cost += cost_weights['deviation']*self.ds[n]*ey**2 # deviation from road desciptor
            
            cost += cost_weights['w']*input[1]**2 # steer angle rate
        
        for n in range(self.N - 1):
            input = self.action[:,n]
            next_input = self.action[:,n+1]
            
            # acceleration continuous
            cost += cost_weights['a']*(next_input[self.car.input.index('a')]-input[self.car.input.index('a')])
        
        cost += ca.if_else(self.state[self.car.state.index('v'),-1] >= state_constraints['v_max'],
            cost_weights['v']*(self.state[self.car.state.index('v'),-1] - state_constraints['v_max'])**2, 0) 
        cost += cost_weights['time']*self.state[self.car.state.index('t'),-1] # final cost (minimize time)
        cost += cost_weights['ey']*self.state[self.car.state.index('ey'),-1]**2 # final cost (minimize terminal lateral error) hardcodato
        cost += cost_weights['epsi']*self.state[self.car.state.index('epsi'),-1]**2 # final cost (minimize terminal course error) hardcodato
        opti.minimize(cost)
            
        # -------------------- Input Constraints ------------------------------------------
        input_constraints = self.config['input_constraints']
        for n in range(self.N): # loop over control intervals
            opti.subject_to(self.action[0,n] <= input_constraints['a_max'])
            opti.subject_to(self.action[0,n] >= input_constraints['a_min'])
            opti.subject_to(self.action[1,n] <= input_constraints['w_max'])
            opti.subject_to(self.action[1,n] >= input_constraints['w_min'])
            
        return opti
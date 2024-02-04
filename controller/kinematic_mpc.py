from model.kinematic_car import KinematicCar, KinematicCarInput
import casadi as ca
import numpy as np
from casadi import cos, sin, tan
from controller.controller import Controller

class KinematicMPC(Controller):
    def __init__(self, car: KinematicCar, config):
        self.dt = config['mpc_dt']
        self.car = car
        self.N = config['horizon']
        self.config = config
        self.ns = len(self.car.state) # number of state variables
        self.na = len(self.car.input) # number of action variables
        self.opti = self._init_opti()
        
    def command(self, state):
        self._init_parameters(state)
        sol = self.opti.solve()
        self.action_prediction = sol.value(self.action)
        self.state_prediction = sol.value(self.state)
        curvature_prediction = sol.value(self.curvature)
        next_input = KinematicCarInput(a=self.action_prediction[0][0], w=self.action_prediction[1][0])
        return next_input, self.state_prediction, self.action_prediction, curvature_prediction
    
    def _init_parameters(self, state):
        self.opti.set_value(self.state0, state)
        self.opti.set_initial(self.action, self.action_prediction)
        self.opti.set_initial(self.state, self.state_prediction)
        
    def _init_opti(self):
        # ========================= Optimizer Initialization =================================
        opti = ca.Opti()
        p_opts = {'ipopt.print_level': 0, 'print_time': False, 'expand': False}
        s_opts = {}
        opti.solver("ipopt", p_opts, s_opts)
        
        # ========================= Decision Variables with Initialization ===================
        self.state = opti.variable(self.ns, self.N+1) # state trajectory var
        self.action = opti.variable(self.na, self.N)   # control trajectory var
        self.state_prediction = np.ones((self.ns, self.N+1))*2 # actual predicted state trajectory
        self.action_prediction = np.zeros((self.na, self.N))   # actual predicted control trajectory
        
        # ======================== Helper Variables ==========================================
        self.state0 = opti.parameter(self.ns) # initial state
        opti.subject_to(self.state[:,0] == self.state0) # constraint on initial state
        self.ds = opti.variable(self.N) #ds trajectory during the planning horizon
        self.curvature = opti.variable(self.N+1)
        opti.subject_to(self.curvature[0] == self.car.track.get_curvature(self.state0[self.car.state.index('s')]))
        
        # ======================= Cycle the entire horizon defining NLP problem ===============
        cost_weights = self.config['cost_weights'] 
        state_constraints = self.config['state_constraints']
        input_constraints = self.config['input_constraints']
        cost = 0
        for n in range(self.N):
            # extracting state and input at current iteration of horizon
            state = self.state[:,n]
            state_next = self.state[:,n+1]
            input = self.action[:,n]
            v = state[self.car.state.index('v')]
            ey = state[self.car.state.index('ey')]
            epsi = state[self.car.state.index('epsi')]
            delta = state[self.car.state.index('delta')]
            a = input[self.car.input.index('a')]
            w = input[self.car.input.index('w')]
            
            # -------------------- Stage Cost -------------------------------------
            cost += ca.if_else(ey < state_constraints['ey_min'], # violation of road bounds
                       cost_weights['boundary']*self.ds[n]*(ey - state_constraints['ey_min'])**2, 0)
            
            cost += ca.if_else(ey > state_constraints['ey_max'], # violation of road bounds
                       cost_weights['boundary']*self.ds[n]*(ey - state_constraints['ey_max'])**2, 0)
                
            cost += cost_weights['deviation']*self.ds[n]*ey**2 # deviation from road desciptor
            
            cost += cost_weights['w']*w**2 # steer angle rate

            # -------------------- Constraints ------------------------------------------
            # state limits
            opti.subject_to(v    >= state_constraints['v_min'])
            opti.subject_to(delta >= state_constraints['delta_min'])
            opti.subject_to(delta <= state_constraints['delta_max'])
            
            # input limits
            opti.subject_to(a >= input_constraints['a_min'])
            opti.subject_to(a <= input_constraints['a_max'])
            opti.subject_to(w >= input_constraints['w_min'])
            opti.subject_to(w <= input_constraints['w_max'])
            
            # dynamics
            opti.subject_to(self.ds[n] == self.dt * ((v * cos(epsi)) / (1 - ey * self.curvature[n]))) # going on for dt and snapshot of how much the car moved
            opti.subject_to(state_next == self.car.spatial_transition(state,input,self.curvature[n],self.ds[n]))
            opti.subject_to(self.curvature[n+1] == self.car.track.get_curvature(state_next[self.car.state.index('s')]))
            
        # ------------------ Stage Cost for Force Input Continuity ----------------------            
        for n in range(self.N - 1):
            input = self.action[:,n]
            next_input = self.action[:,n+1]
            # acceleration continuous
            cost += cost_weights['a']*(next_input[self.car.input.index('a')]-input[self.car.input.index('a')])
        
        # -------------------- Terminal Cost -----------------------
        cost += ca.if_else(self.state[self.car.state.index('v'),-1] >= state_constraints['v_max'], # excessive speed
            cost_weights['v']*(self.state[self.car.state.index('v'),-1] - state_constraints['v_max'])**2, 0) 
        cost += cost_weights['time']*(self.state[self.car.state.index('t'),-1]-self.state[self.car.state.index('t'),0]) # final cost (minimize time)
        cost += cost_weights['ey']*self.state[self.car.state.index('ey'),-1]**2 # final cost (minimize terminal lateral error) hardcodato
        cost += cost_weights['epsi']*self.state[self.car.state.index('epsi'),-1]**2 # final cost (minimize terminal course error) hardcodato
        
        opti.minimize(cost)
        return opti
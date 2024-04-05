from models.kinematic_car import KinematicCar, KinematicInput
import casadi as ca
import numpy as np
from casadi import cos, sin, tan
from controllers.controller import Controller
np.random.seed(3)

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
        self._init_horizon(state)
        sol = self.opti.solve()
        self.action_prediction = sol.value(self.action)
        self.state_prediction = sol.value(self.state)
        return KinematicInput(a=self.action_prediction[0][0], w=self.action_prediction[1][0])
    
    def _init_horizon(self, state):
        # ======================= Initializing state and action prediction ============================
        state = state.values.squeeze()
        self.opti.set_value(self.state0, state)
        self.opti.set_initial(self.action, self.action_prediction)
        self.opti.set_initial(self.state, self.state_prediction)
        
    def _init_opti(self):
        # ========================= Optimizer Initialization =================================
        opti = ca.Opti('nlp')
        ipopt_options = {'print_level': 0, 'linear_solver': 'ma27', 'hsllib': '/usr/local/lib/libcoinhsl.so', 'fixed_variable_treatment': 'relax_bounds'}
        options = {'print_time': False, 'expand': False, 'ipopt': ipopt_options}
        opti.solver("ipopt", options)
        
        # ========================= Decision Variables with Initialization ===================
        self.state = opti.variable(self.ns, self.N+1) # state trajectory var
        self.action = opti.variable(self.na, self.N)   # control trajectory var
        self.state_prediction = np.random.random((self.ns, self.N+1)) # actual predicted state trajectory
        self.action_prediction = np.random.random((self.na, self.N))   # actual predicted control trajectory
        self.state0 = opti.parameter(self.ns) # initial state
        opti.subject_to(self.state[:,0] == self.state0) # constraint on initial state
        
        # ======================= Cycle the entire horizon defining NLP problem ===============
        cost_weights = self.config['cost_weights'] 
        state_constraints = self.config['state_constraints']
        input_constraints = self.config['input_constraints']
        cost = 0
        
        for n in range(self.N):
            # -------- State and Input Extraction --------------------------------
            state = self.state[:,n]
            state_next = self.state[:,n+1]
            input = self.action[:,n]
            v = state[self.car.state.index('v')]
            ey = state[self.car.state.index('ey')]
            epsi = state[self.car.state.index('epsi')]
            delta = state[self.car.state.index('delta')]
            a = input[self.car.input.index('a')]
            w = input[self.car.input.index('w')]
            
            # ---------- Discretization and model dynamics ------------------------
            # going on for dt and snapshot of how much the car moved
            curvature = self.car.track.get_curvature(state_next[self.car.state.index('s')])
            ds = self.dt * ((v * cos(epsi)) / (1 - curvature*ey))
            opti.subject_to(state_next == self.car.spatial_transition(state,input,curvature,ds))
            
            # -------------------- Stage Cost -------------------------------------
            cost += cost_weights['deviation']*ds*ey**2 # deviation from road desciptor
            
            cost += ca.if_else(ey < state_constraints['ey_min'], # violation of road bounds
                       cost_weights['boundary']*ds*(ey - state_constraints['ey_min'])**2, 0)
            
            cost += ca.if_else(ey > state_constraints['ey_max'], # violation of road bounds
                       cost_weights['boundary']*ds*(ey - state_constraints['ey_max'])**2, 0)
            
            cost += cost_weights['w']*w**2 #steer angle rate
            
            if n < self.N-1: #Input Continuity
                next_input = self.action[:,n+1]
                cost += cost_weights['a']*(1/v)*(next_input[self.car.input.index('a')]-input[self.car.input.index('a')])**2 

            # -------------------- Constraints ------------------------------------------
            # state bounds
            opti.subject_to(v >= state_constraints['v_min'])
            opti.subject_to(opti.bounded(state_constraints['delta_min'],delta,state_constraints['delta_max']))
            
            # input bounds
            opti.subject_to(opti.bounded(input_constraints['a_min'],a,input_constraints['a_max']))
            opti.subject_to(opti.bounded(input_constraints['w_min'],w,input_constraints['w_max']))
            
        # -------------------- Terminal Cost -----------------------
        cost += ca.if_else(self.state[self.car.state.index('v'),-1] >= state_constraints['v_max'], # excessive speed
            cost_weights['v']*(self.state[self.car.state.index('v'),-1] - state_constraints['v_max'])**2, 0) 
        cost += cost_weights['time']*self.state[self.car.state.index('t'),-1]  # final cost (minimize time)
        cost += cost_weights['ey']*self.state[self.car.state.index('ey'),-1]**2 # final cost (minimize terminal lateral error)
        cost += cost_weights['epsi']*self.state[self.car.state.index('epsi'),-1]**2 # final cost (minimize terminal course error)
        
        opti.minimize(cost)
        return opti
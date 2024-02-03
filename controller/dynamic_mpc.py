from model.dynamic_car import DynamicCar, DynamicCarInput
import casadi as ca
import numpy as np
from casadi import cos, sin, tan, fabs
from controller.controller import Controller

class DynamicMPC(Controller):
    def __init__(self, car: DynamicCar, config):
        self.dt = config['mpc_dt']
        self.car = car
        self.N = config['horizon']
        self.config = config
        self.ns = len(self.car.state) # number of state variables
        self.na = len(self.car.input) # number of action variables
        self.opti = self._init_opti()
        
    def command(self, state):
        # every new horizon, the current state (and the last prediction) is the initial 
        self._init_parameters(state)
        sol = self.opti.solve()
        self.action_prediction = sol.value(self.action)
        self.state_prediction = sol.value(self.state)
        curvature_prediction = sol.value(self.curvature)
        next_input = DynamicCarInput(Fx=self.action_prediction[0][0], w=self.action_prediction[1][0])
        # next_input = DynamicCarInput(Fx = 500, w = 0)
        return next_input, self.state_prediction, self.action_prediction, curvature_prediction
    
    def _init_parameters(self, state):
        '''
        s_k is a state in its entirety. I need to extract relevant variables
        '''
        self.opti.set_value(self.state0, state)
        self.opti.set_initial(self.action, self.action_prediction)
        self.opti.set_initial(self.state, self.state_prediction)
        
    def _init_opti(self):
        # -------------------- Optimizer Initialization ----------------------------------
        opti = ca.Opti()
        p_opts = {'ipopt.print_level': 0, 'print_time': False, 'expand': False}
        s_opts = {}
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
        self.curvature = opti.variable(self.N+1)
        opti.subject_to(self.curvature[0] == self.car.track.get_curvature(self.state0[self.car.state.index('s')]))

        # --------------------- Slack Variables ------------------------------------------
        self.Fe = opti.variable(2, self.N)      # Slack variables for excessive force usage beyond the imposed limits
                                                # Fe[0] for front tire, Fe[1] for rear tire

        # -------------------- Model Constraints ------------------------------------------
        state_constraints = self.config['state_constraints']
        for n in range(self.N):
            state = self.state[:,n]
            state_next = self.state[:,n+1]
            input = self.action[:,n]
            
            # state constraints
            opti.subject_to(state[self.car.state.index('Ux')] >= state_constraints['Ux_min']) # TODO without this, things break
            opti.subject_to(state[self.car.state.index('delta')] <= state_constraints['delta_max'])
            opti.subject_to(state[self.car.state.index('delta')] >= state_constraints['delta_min'])
            
            # continuity contraint on spatial dynamics
            Ux = state[self.car.state.index('Ux')]
            Uy = state[self.car.state.index('Uy')]
            ey = state[self.car.state.index('ey')]
            epsi = state[self.car.state.index('epsi')]
            opti.subject_to(self.ds[n] == self.dt * (Ux*ca.cos(epsi) - Uy*ca.sin(epsi)) / (1 - self.curvature[n]*ey)) # going on for dt and snapshot of how much the car moved
            opti.subject_to(state_next == self.car.spatial_transition(state,input,self.curvature[n],self.ds[n]))
            opti.subject_to(self.curvature[n+1] == self.car.track.get_curvature(state_next[self.car.state.index('s')]))

        # ------------------- Cost Function --------------------------------------

        cost_weights = self.config['cost_weights'] 
        cost = 0
        for n in range(self.N):
            state = self.state[:,n]
            input = self.action[:,n]
            ey = state[self.car.state.index('ey')]
            Ux = state[self.car.state.index('Ux')]
            Uy = state[self.car.state.index('Uy')]
            delta = state[self.car.state.index('delta')]
            r = state[self.car.state.index('r')]

            Fx = input[self.car.input.index('Fx')]
            
            # violation of road bounds
            cost += ca.if_else(ey < state_constraints['ey_min'],
                       cost_weights['boundary']*self.ds[n]*(ey - state_constraints['ey_min'])**2, 0)
            
            cost += ca.if_else(ey > state_constraints['ey_max'],
                       cost_weights['boundary']*self.ds[n]*(ey - state_constraints['ey_max'])**2, 0)
                
            cost += cost_weights['deviation']*self.ds[n]*ey**2 # deviation from road desciptor
            
            cost += cost_weights['w']*input[1]**2 # steer angle rate

            cost += ca.if_else(fabs(tan(self.car.alpha_f(Ux,Uy,r,delta))) >= tan(self.car.alphamod_f(Fx)),  # slip angle front
                        cost_weights['slip']*(fabs(tan(self.car.alpha_f(Ux,Uy,r,delta))) - tan(self.car.alphamod_f(Fx)))**2, 0)
            
            cost += ca.if_else(fabs(tan(self.car.alpha_r(Ux,Uy,r,delta))) >= tan(self.car.alphamod_r(Fx)),  # slip angle rear
                        cost_weights['slip']*(fabs(tan(self.car.alpha_r(Ux,Uy,r,delta))) - tan(self.car.alphamod_r(Fx)))**2, 0)
            
            cost += cost_weights['friction']*((self.Fe[0, n]**2)**2 + (self.Fe[1, n]**2)**2)

        for n in range(self.N - 1):
            input = self.action[:,n]
            next_input = self.action[:,n+1]
            
            # acceleration continuous
            cost += cost_weights['Fx']*(next_input[self.car.input.index('Fx')]-input[self.car.input.index('Fx')])
        
        cost += ca.if_else(self.state[self.car.state.index('Ux'),-1] >= state_constraints['Ux_max'],
            cost_weights['Ux']*(self.state[self.car.state.index('Ux'),-1] - state_constraints['Ux_max'])**2, 0) 
        cost += cost_weights['time']*self.state[self.car.state.index('t'),-1] # final cost (minimize time)
        cost += cost_weights['ey']*self.state[self.car.state.index('ey'),-1]**2 # final cost (minimize terminal lateral error) hardcodato
        cost += cost_weights['epsi']*self.state[self.car.state.index('epsi'),-1]**2 # final cost (minimize terminal course error) hardcodato
        opti.minimize(cost)
            
        # -------------------- Input Constraints TODO ------------------------------------------
        input_constraints = self.config['input_constraints']
        for n in range(self.N): # loop over control intervals
            opti.subject_to(self.action[0,n] <= input_constraints['Fx_max'])
            opti.subject_to(self.action[0,n] >= input_constraints['Fx_min'])
            opti.subject_to(self.action[1,n] <= input_constraints['w_max'])
            opti.subject_to(self.action[1,n] >= input_constraints['w_min'])

        # -------------------- Tire model and friction limit constraints ------------------------------------------
        
        for n in range(self.N):
            state = self.state[:,n]
            input = self.action[:,n]

            Fx = input[self.car.input.index('Fx')]
            Ux = self.state[self.car.state.index('Ux')]
            Uy = self.state[self.car.state.index('Uy')]
            delta = self.state[self.car.state.index('delta')]
            r = self.state[self.car.state.index('r')]
            
            mu = self.car.config['env']['mu']
            
            # opti.subject_to(self.car.Fx_f(Fx) >= -mu['f']*self.car.Fz_f(Ux, Fx)*cos(self.car.alpha_f(Ux,Uy,r,delta)))
            # opti.subject_to(self.car.Fx_f(Fx) <=  mu['f']*self.car.Fz_f(Ux, Fx)*cos(self.car.alpha_f(Ux,Uy,r,delta)))
            
            # opti.subject_to(self.car.Fx_r(Fx) >= -mu['r']*self.car.Fz_r(Ux, Fx)*cos(self.car.alpha_r(Ux,Uy,r,delta)))
            # opti.subject_to(self.car.Fx_r(Fx) <= mu['r']*self.car.Fz_r(Ux, Fx)*cos(self.car.alpha_r(Ux,Uy,r,delta)))
            
            # opti.subject_to(self.car.Fx_f(Fx)**2 + (self.car.Fy_f(Ux, Uy, r, delta, Fx))**2 <= (mu['r']*self.car.Fz_r(Ux, Fx))**2 + (self.Fe[0])**2) 

        return opti
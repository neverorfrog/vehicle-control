from model.dynamic_car import DynamicCar, DynamicCarInput
import casadi as ca
import numpy as np
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
        
    def command(self, state, curvature):
        # every new horizon, the current state (and the last prediction) is the initial 
        print("") # to separate between prints
        self._init_parameters(state, curvature)
        sol = self.opti.solve()
        self.action_prediction = sol.value(self.action)
        self.state_prediction = sol.value(self.state)
        curvature = sol.value(self.curvature)
        # next_input = DynamicCarInput(Fx=self.action_prediction[0][0], w=self.action_prediction[1][0])
        next_input = DynamicCarInput(Fx = 0, w = 0)
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
        p_opts = {'ipopt.print_level': 0, 'print_time': False, 'expand': False}
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

        # --------------------- Slack Variables ------------------------------------------
        self.Fe = opti.variable(2, self.N)      # Slack variables for excessive force usage beyond the imposed limits
                                                # Fe[0] for front tire, Fe[1] for rear tire

        # -------------------- Model Constraints TODO ------------------------------------------
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
            opti.subject_to(self.s[n+1] == self.s[n] + self.ds[n])
            opti.subject_to(self.curvature[n] == self.car.curvature(self.s[n]))
            opti.subject_to(state_next == self.car.spatial_transition(state,input,self.curvature[n],self.ds[n]))



        # ------------------- Cost Function TODO --------------------------------------
      
              
        p = self.car.get_car_parameters()
        g,theta,phi,Av2,Crr,eps,mu,C_alpha,Cd = self.car.get_parameters()
        
        print("#############  ", p.m,"  ", p.a,"  ", p.b,"  ", p.h_cg,"  ", p.Izz)
        
        print("#############    ", Ux)
        Ux_sym = ca.SX.sym('Ux_sym')
        Uy_sym = ca.SX.sym('Ux_sym')
        delta_sym = ca.SX.sym('Ux_sym')
        r_sym = ca.SX.sym('Ux_sym')
        p_sym = ca.SX.sym('Ux_sym', len(p))
        Fx_sym = ca.SX.sym('Fx_sym')
        
        Xf, Xr = self.car._get_force_distribution(Fx_sym)
        Fz_r = self.car.get_Fz_r_function(Ux_sym, Fx_sym, Xf, p, p.a+p.b, g, theta, phi, Av2)
        Fz_f = self.car.get_Fz_f_function(Ux_sym, Fx_sym, Xf, p, p.a+p.b, g, theta, phi, Av2)
        Fy_max = self.car.get_Fy_max_function(Ux_sym, Fx_sym,  Fz_f, Fz_r, Xf, mu)
        
        alpha_f = self.car.get_alpha_f_function(Uy_sym, Ux_sym, delta_sym, r_sym, p)
        alpha_r = self.car.get_alpha_r_function(Uy_sym, Ux_sym, r_sym, p)
        alpha_mod = self.car.get_alpha_mod_function(Ux_sym, Fx_sym, Fy_max, eps, C_alpha)
       
        Fy = self.car._get_lateral_force(alpha_f, alpha_r, Uy_sym, Ux_sym, delta_sym, r_sym, Fz_f, Fz_r, Fx_sym, Xf, eps, mu, C_alpha)



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

            cost += ca.if_else(ca.fabs(ca.tan(alpha_f(Uy, Ux, delta, r))) >= ca.tan(alpha_mod(Ux, Fx)[0]),  # slip angle front
                        cost_weights['slip']*(ca.fabs(ca.tan(alpha_f(Uy, Ux, delta, r))) - ca.tan(alpha_mod(Ux,Fx)[0]))**2, 0)
            
            cost += ca.if_else(ca.fabs(ca.tan(alpha_r(Uy, Ux, r))) >= ca.tan(alpha_mod(Ux, Fx)[1]),         # slip angle rear
                        cost_weights['slip']*(ca.fabs(ca.tan(alpha_r(Uy, Ux, r))) - ca.tan(alpha_mod(Ux,Fx)[1]))**2, 0)
            
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

        # -------------------- Tire model Constraints TODO ------------------------------------------
        
        for n in range(self.N):
            state = self.state[:,n]
            input = self.action[:,n]

            Xf,Xf = self.car._get_force_distribution(Fx_sym)
            Fx = input[self.car.input.index('Fx')]
            Ux = self.state[self.car.state.index('Ux')]
            Uy = self.state[self.car.state.index('Uy')]
            delta = self.state[self.car.state.index('delta')]
            r = self.state[self.car.state.index('r')]
            
            opti.subject_to(Xf(Fx)*Fx >= -mu*Fz_f(Ux, Fx)*ca.cos(alpha_f(Uy, Ux, delta, r)))
            opti.subject_to(Xf(Fx)*Fx <= mu*Fz_f(Ux, Fx)*ca.cos(alpha_f(Uy, Ux, delta, r)))

            opti.subject_to(Xr(Fx)*Fx >= -mu*Fz_r(Ux, Fx)*ca.cos(alpha_r(Uy, Ux, r)))
            opti.subject_to(Xr(Fx)*Fx <= mu*Fz_r(Ux, Fx)*ca.cos(alpha_r(Uy, Ux, r)))
            
        # -------------------- Friction limits Constraints TODO ------------------------------------------
        # some mu must be mu_limit
        for n in range(self.N): 
            state = self.state[:,n]
            input = self.action[:,n]
            Xf,Xf = self.car._get_force_distribution(Fx_sym)
            Fx = input[self.car.input.index('Fx')]
            Ux = self.state[self.car.state.index('Ux')]
            Uy = self.state[self.car.state.index('Uy')]
            delta = self.state[self.car.state.index('delta')]
            r = self.state[self.car.state.index('r')]

            opti.subject_to((Xf(Fx)*Fx)**2 + (Fy(Uy, Ux, delta, r, Fx)[0])**2 <= (mu*Fz_f(Ux, Fx))**2 + (self.Fe[0])**2)
            # opti.subject_to((Xr(Fx)*Fx)**2 + (Fy(Uy, Ux, delta, r, Fx)[1])**2 <= (mu*Fz_r(Ux, Fx))**2 + (self.Fe[1])**2)


        return opti
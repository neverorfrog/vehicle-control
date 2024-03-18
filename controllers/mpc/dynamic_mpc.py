from models.dynamic_car import DynamicCar, DynamicCarInput
import casadi as ca
import numpy as np
from casadi import cos, sin, tan, fabs
from controllers.controller import Controller
np.random.seed(30)

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
        self._init_horizon(state)
        sol = self.opti.solve()
        self.action_prediction = sol.value(self.action)
        self.state_prediction = sol.value(self.state)
        print(f"Solver iterations: {sol.stats()["iter_count"]}")
        print(f"ds: {sol.value(self.ds)}")
        return DynamicCarInput(Fx=self.action_prediction[0][0], w=self.action_prediction[1][0])
    
    def _init_horizon(self, state):
        state = state.values.squeeze()
        self.opti.set_value(self.state0, state)
        self.opti.set_initial(self.action, self.action_prediction)
        self.opti.set_initial(self.state, self.state_prediction)
        
    def _init_opti(self):
        """
        Optimizer Initialization
                
        A function that initializes the optimizer and the decision variables, and defines the NLP problem.

        Returns:
        - opti: The initialized optimizer
        """
        # ========================= Optimizer Initialization =================================
        opti = ca.Opti()
        p_opts = {'ipopt.print_level': 0, 'print_time': False, 'expand': False}
        s_opts = {'fixed_variable_treatment': 'relax_bounds', 'linear_solver': 'ma27','hsllib': "/home/flavio/Programs/hsl/lib/libcoinhsl.so"}
        opti.solver("ipopt", p_opts, s_opts)
        
        # ========================= Decision Variables with Initialization ===================
        self.state = opti.variable(self.ns, self.N+1) # state trajectory var
        self.action = opti.variable(self.na, self.N)   # control trajectory var
        self.state_prediction = np.ones((self.ns, self.N+1)) # actual predicted state trajectory
        self.action_prediction = np.ones((self.na, self.N))# actual predicted control trajectory
        self.ds = opti.variable(self.N) # ds trajectory var
        self.state0 = opti.parameter(self.ns) # initial state
        opti.subject_to(self.state[:,0] == self.state0) # constraint on initial state

        # ======================== Slack Variables ============================================
        self.Fe_f = opti.variable(self.N) 
        self.Fe_r = opti.variable(self.N)  
        # self.Fx_f = opti.variable(self.N) 
        # self.Fx_r = opti.variable(self.N)
        # self.Fy_f = opti.variable(self.N) 
        # self.Fy_r = opti.variable(self.N)
                                           
        # ======================= Cycle the entire horizon defining NLP problem ===============
        cost_weights = self.config['cost_weights'] 
        state_constraints = self.config['state_constraints']
        input_constraints = self.config['input_constraints']
        Peng = self.car.config['car']['Peng']
        mu = self.car.config['env']['mu']
        cost = 0
        for n in range(self.N):
            # ----------- Extracting State and Input --------------------------------------------
            state = self.state[:,n]
            state_next = self.state[:,n+1]
            input = self.action[:,n]
            Ux = state[self.car.state.index('Ux')]
            Uy = state[self.car.state.index('Uy')]
            ey = state[self.car.state.index('ey')]
            epsi = state[self.car.state.index('epsi')]
            r = state[self.car.state.index('r')]
            delta = state[self.car.state.index('delta')]
            Fx = input[self.car.input.index('Fx')]
            w = input[self.car.input.index('w')]
            
            # -------------------- Constraints --------------------------------------------------
            # state limits
            opti.subject_to(Ux >= state_constraints['Ux_min'])
            opti.subject_to(opti.bounded(state_constraints['delta_min'],delta,state_constraints['delta_max']))
            
            # Discretization (Going on for dt with displacement snapshot) 
            curvature = self.car.track.get_curvature(state[self.car.state.index('s')])
            ds = self.dt * Ux * cos(epsi)
            opti.subject_to(self.ds[n] == ds)
            
            # Model dynamics 
            opti.subject_to(state_next == self.car.spatial_transition(state,input,curvature,self.ds[n])) 
                
            # input limits
            opti.subject_to(Fx <= Peng / Ux)
            opti.subject_to(opti.bounded(input_constraints['w_min'],w,input_constraints['w_max']))
            
            # longitudinal force limits on tires
            bound_f = mu['f']*self.car.Fz_f(Ux,Fx)*cos(self.car.alpha_f(Ux,Uy,r,delta))
            opti.subject_to(opti.bounded(-bound_f,self.car.Fx_f(Fx),bound_f))
            bound_r = mu['r']*self.car.Fz_r(Ux,Fx)*cos(self.car.alpha_r(Ux,Uy,r,delta))
            opti.subject_to(opti.bounded(-bound_r,self.car.Fx_r(Fx),bound_r))   
            
            # -------------------- Stage Cost -------------------------------------------
            cost += ca.if_else(ey < state_constraints['ey_min'], # violation of road bounds
                       cost_weights['boundary']*ds*(ey - state_constraints['ey_min'])**2, 0)
            
            cost += ca.if_else(ey > state_constraints['ey_max'], # violation of road bounds
                       cost_weights['boundary']*ds*(ey - state_constraints['ey_max'])**2, 0)
                
            cost += cost_weights['deviation']*ds*(ey**2) # deviation from road desciptor
            
            cost += cost_weights['w']*(w**2) # steer angle rate

            cost += ca.if_else(fabs(tan(self.car.alpha_f(Ux,Uy,r,delta))) >= tan(self.car.alphamod_f(Fx)),  # slip angle front
                        cost_weights['slip']*(fabs(tan(self.car.alpha_f(Ux,Uy,r,delta))) - tan(self.car.alphamod_f(Fx)))**2, 0)
            
            cost += ca.if_else(fabs(tan(self.car.alpha_r(Ux,Uy,r,delta))) >= tan(self.car.alphamod_r(Fx)),  # slip angle rear
                        cost_weights['slip']*(fabs(tan(self.car.alpha_r(Ux,Uy,r,delta))) - tan(self.car.alphamod_r(Fx)))**2, 0)
              
            cost += cost_weights['friction']*((self.Fe_f[n]**2)**2 + (self.Fe_r[n]**2)**2) # slack variables for sparsity
                      
            if n < self.N-1: #Force Input Continuity
                next_input = self.action[:,n+1]
                Fx_next = next_input[self.car.input.index('Fx')]
                cost += cost_weights['Fx']*(1/ds)*(Fx_next - Fx)**2 
            
            # --------- Stuff that breaks things (Friction Limits for real world experiments) -----
            # opti.subject_to(self.Fy_f[n] == self.car.Fy_f(Ux,Uy,r,delta,Fx))
            # opti.subject_to(self.Fy_r[n] == self.car.Fy_r(Ux,Uy,r,delta,Fx))
            # opti.subject_to(self.Fx_f[n] == self.car.Fx_f(Fx))
            # opti.subject_to(self.Fx_r[n] == self.car.Fx_r(Fx))
            # opti.subject_to(self.Fx_f[n]**2 + self.Fy_f[n]**2 <= (input_constraints['mu_lim']*self.car.Fz_f(Ux,Fx))**2 + (self.Fe_f[n])**2)
            # opti.subject_to(self.Fx_f[n]**2 + self.Fy_r[n]**2 <= (input_constraints['mu_lim']*self.car.Fz_r(Ux,Fx))**2 + (self.Fe_r[n])**2) 
            # ---------------------------------------------------------------------
        
        # -------------------- Terminal Cost -----------------------
        cost += ca.if_else(self.state[self.car.state.index('Ux'),-1] >= state_constraints['Ux_max'], # excessive speed
            cost_weights['Ux']*(self.state[self.car.state.index('Ux'),-1] - state_constraints['Ux_max'])**2, 0) 
        cost += cost_weights['time']*self.state[self.car.state.index('t'),-1] # final cost (minimize time)
        cost += cost_weights['ey']*self.state[self.car.state.index('ey'),-1]**2 # final cost (minimize terminal lateral error) hardcodato
        cost += cost_weights['epsi']*self.state[self.car.state.index('epsi'),-1]**2 # final cost (minimize terminal course error) hardcodato
            
        opti.minimize(cost)
        return opti
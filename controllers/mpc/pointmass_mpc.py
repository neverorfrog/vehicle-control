from models.dynamic_car import DynamicCar, DynamicCarInput
import casadi as ca
import numpy as np
from casadi import cos, sin, tan, fabs
from controllers.controller import Controller
np.random.seed(3)

# TODO

class PointMassMPC(Controller):
    def __init__(self, car: DynamicCar, config):
        self.dt = config['mpc_dt']
        self.car = car
        self.N = config['horizon']
        self.config = config
        self.ns = len(self.car.state) # number of state variables
        self.na = len(self.car.input) # number of action variables
        self.opti = self._init_opti()
        self._init_racing()
        
    def command(self, state):
        self._init_horizon(state)
        sol = self.opti.solve()
        self.action_prediction = sol.value(self.action)
        self.state_prediction = sol.value(self.state)
        next_input = DynamicCarInput(Fx=self.action_prediction[0][0], w=self.action_prediction[1][0])
        return next_input, self.state_prediction, self.action_prediction
    
    def _init_racing(self):
        '''Initializing state prediction'''
        pass
    
    def _init_horizon(self, state):
        state = state.values.squeeze()
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
        self.state_prediction = np.ones((self.ns, self.N+1)) # actual predicted state trajectory
        self.action_prediction = np.zeros((self.na, self.N))   # actual predicted control trajectory
        self.state0 = opti.parameter(self.ns) # initial state
        opti.subject_to(self.state[:,0] == self.state0) # constraint on initial state

        # ======================== Slack Variables ============================================
        self.Fy_f = opti.variable(self.N)
        self.Fy_r = opti.variable(self.N)
        self.Fe_f = opti.variable(self.N)
        self.Fe_r = opti.variable(self.N)  
                                           
        # ======================= Cycle the entire horizon defining NLP problem ===============
        cost_weights = self.config['cost_weights'] 
        state_constraints = self.config['state_constraints']
        input_constraints = self.config['input_constraints']
        Peng = self.car.config['car']['Peng']
        mu = self.car.config['env']['mu']
        cost = 0
        for n in range(self.N):
            # extracting state and input at current iteration of horizon
            state = self.state[:,n]
            state_next = self.state[:,n+1]
            input = self.action[:,n]
            V = state[self.car.state.index('V')]
            ey = state[self.car.state.index('ey')]
            epsi = state[self.car.state.index('epsi')]
            Fx = input[self.car.input.index('Fx')]
            Fy = input[self.car.input.index('Fy')]
            
            # ---------- Discretization and model dynamics ------------------------
            # going on for dt and snapshot of how much the car moved
            curvature = self.car.track.get_curvature(state[self.car.state.index('s')]) #departing from s=0 we get NaN values due to the curvature at s=0 (dunno why)
            ds = self.dt * V*ca.cos(epsi)/(1-curvature*ey)
            opti.subject_to(state_next == self.car.spatial_transition(state,input,curvature,ds))
            
            # -------------------- Stage Cost -------------------------------------
            cost += ca.if_else(ey < state_constraints['ey_min'], # violation of road bounds
                       cost_weights['boundary']*ds*(ey - state_constraints['ey_min'])**2, 0)
            
            cost += ca.if_else(ey > state_constraints['ey_max'], # violation of road bounds
                       cost_weights['boundary']*ds*(ey - state_constraints['ey_max'])**2, 0)
                
            cost += cost_weights['deviation']*ds*(ey**2) # deviation from road desciptor
            
            cost += cost_weights['friction']*((self.Fe_f[n]**2)**2 + (self.Fe_r[n]**2)**2) # friction limit
            
            if n < self.N-1: #Force Input Continuity
                next_input = self.action[:,n+1]
                Fx_next = next_input[self.car.input.index('Fx')]
                Fy_next = next_input[self.car.input.index('Fy')]
                cost += cost_weights['Fx']*(1/ds)*(Fx_next-Fx)**2 
                cost += cost_weights['Fx']*(1/ds)*(Fy_next-Fy)**2 
            
            # -------------------- Constraints ------------------------------------------
            # state limits
            opti.subject_to(V >= state_constraints['V_min'])
            
            # input limits
            opti.subject_to(Fx <= Peng / V)
            
            # friction limits
            # opti.subject_to((Fx*self.car.Xf(Fx))**2 + self.Fy_f[n]**2 <= (input_constraints['mu_lim']*self.car.Fz_f(Ux,Fx))**2 + (self.Fe_f[n])**2)
            # opti.subject_to((Fx*self.car.Xr(Fx))**2 + self.Fy_r[n]**2 <= (input_constraints['mu_lim']*self.car.Fz_r(Ux,Fx))**2 + (self.Fe_r[n])**2) 
        
        # -------------------- Terminal Cost -----------------------
        cost += ca.if_else(self.state[self.car.state.index('V'),-1] >= state_constraints['V_max'], # excessive speed
            cost_weights['V']*(self.state[self.car.state.index('V'),-1] - state_constraints['V_max'])**2, 0) 
        cost += cost_weights['time']*self.state[self.car.state.index('t'),-1] # final cost (minimize time)
        cost += cost_weights['ey']*self.state[self.car.state.index('ey'),-1]**2 # final cost (minimize terminal lateral error) hardcodato
        cost += cost_weights['epsi']*self.state[self.car.state.index('epsi'),-1]**2 # final cost (minimize terminal course error) hardcodato
            
        opti.minimize(cost)
        return opti
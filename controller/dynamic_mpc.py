from model.kinematic_car import KinematicCar
import casadi as ca
import numpy as np
from casadi import cos, sin, tan
from controller.controller import Controller
from model.state import KinematicCarInput
from utils.utils import integrate

class RacingMPC(Controller):
    def __init__(self, car: KinematicCar, config):
        self.dt = config['mpc_dt']
        self.car = car
        self.N = config['horizon']
        self.config = config
        self.ns, self.ni, self.transition = self._init_ode()
        self.opti = self._init_opti()
        
    def command(self, s_k, kappa):
        # every new horizon, the current state (and the last prediction) is the initial 
        print("") # to separate between prints
        self._init_parameters(s_k, kappa)
        sol = self.opti.solve()
        self.action_prediction = sol.value(self.action)
        self.state_prediction = sol.value(self.state)
        ds_traj = sol.value(self.ds)
        next_input = KinematicCarInput(a=self.action_prediction[0][0], w=self.action_prediction[1][0])
        return next_input, self.state_prediction, ds_traj
    
    def _init_parameters(self, s_k, kappa):
        '''
        s_k is a state in its entirety. I need to extract relevant variables
        '''
        state = np.array([s_k.x, s_k.y, s_k.v, s_k.psi, s_k.delta, s_k.ey, s_k.epsi, s_k.t])
        self.opti.set_value(self.kappa, kappa)
        self.opti.set_value(self.s0, state)
        self.opti.set_initial(self.action, self.action_prediction)
        self.opti.set_initial(self.state, self.state_prediction)
        
    def _init_opti(self):
        # -------------------- Optimizer Initialization ----------------------------------
        opti = ca.Opti()
        p_opts = {"ipopt.print_level": 0,"expand":False}
        s_opts = {"max_iter": 100}
        opti.solver("ipopt", p_opts, s_opts)
        
        # -------------------- Decision Variables with Initialization ---------------------
        self.state = opti.variable(self.ns, self.N+1) # predicted state trajectory var
        self.action = opti.variable(self.ni, self.N)   # predicted control trajectory var
        self.state_prediction = np.ones((self.ns, self.N+1)) # actual predicted state trajectory
        self.action_prediction = np.ones((self.ni, self.N))   # actual predicted control trajectory
        self.ds = opti.variable(self.N)
        
        # --------------------- Helper Variables ------------------------------------------
        self.s0 = opti.parameter(self.ns) # initial state
        opti.subject_to(self.state[:,0] == self.s0) # constraint on initial state
        self.kappa = opti.parameter(self.N) # local curvature
        
        # -------------------- Model Constraints ------------------------------------------
        state_constraints = self.config['state_constraints']
        
        for n in range(self.N):
            state = self.state[:,n]
            state_next = self.state[:,n+1]
            input = self.action[:,n]
            
            # state constraints
            opti.subject_to(state[2] >= state_constraints['v_min']) # TODO without this, things break
            opti.subject_to(state[4] <= state_constraints['delta_max'])
            opti.subject_to(state[4] >= state_constraints['delta_min'])
            
            # continuity contraint on spatial dynamics
            v = state[2]
            ey = state[5] # TODO hardcodato
            epsi = state[6]
            opti.subject_to(self.ds[n] == self.dt * ((v * np.cos(epsi)) / (1 - ey * self.kappa[n]))) # going on for dt and snapshot of how much the car moved
            opti.subject_to(state_next == self.transition(state,input,self.kappa[n],self.ds[n]))
            
        # ------------------- Cost Function --------------------------------------
            
        cost_weights = self.config['cost_weights'] 
        cost = 0
        for n in range(self.N):
            state = self.state[:,n]
            input = self.action[:,n]
            ey = state[5] # TODO hardcodato
            
            # violation of road bounds
            cost += ca.if_else(ey < state_constraints['ey_min'],
                       cost_weights['boundary']*(ey - state_constraints['ey_min'])**2, 0)
            
            cost += ca.if_else(ey > state_constraints['ey_max'],
                       cost_weights['boundary']*(ey - state_constraints['ey_max'])**2, 0)
                
            cost += cost_weights['deviation']*self.ds[n]*ey**2 # deviation from road desciptor
            
            cost += cost_weights['w']*input[1]**2 # steer angle rate
        
        for n in range(self.N - 1):
            input = self.action[:,n]
            next_input = self.action[:,n+1]
            
            # acceleration continuous
            cost += cost_weights['a']*(next_input[0]-input[0])
           
        cost += ca.if_else(self.state[2,-1] >= state_constraints['v_max'],
            cost_weights['v']*(self.state[2,-1] - state_constraints['v_max'])**2, 0) 
        cost += cost_weights['time']*self.state[-1,-1] # final cost (minimize time)
        cost += cost_weights['ey']*self.state[5,-1]**2 # final cost (minimize terminal lateral error)
        cost += cost_weights['epsi']*self.state[6,-1]**2 # final cost (minimize terminal course error)
        opti.minimize(cost)
            
        # -------------------- Input Constraints ------------------------------------------
        input_constraints = self.config['input_constraints']
        for n in range(self.N): # loop over control intervals
            opti.subject_to(self.action[0,n] <= input_constraints['a_max'])
            opti.subject_to(self.action[0,n] >= input_constraints['a_min'])
            opti.subject_to(self.action[1,n] <= input_constraints['w_max'])
            opti.subject_to(self.action[1,n] >= input_constraints['w_min'])
            
        return opti
            
    def _init_ode(self):
        '''Differential equations describing the model inside the prediction horizon'''
        ns = 8 # number of states -> (v,psi,delta,ey,epsi,t)
        ni = 2 # number of inputs -> (a,w)
        
        # input variables
        a = ca.SX.sym('a') # driving acceleration
        w = ca.SX.sym('w') # steering angle rate
        input = ca.vertcat(a,w)
        
        # state and auxiliary variables
        x,y,v,psi,delta,s,ey,epsi,t = self.car.state.variables
        kappa = ca.SX.sym('kappa')
        ds = ca.SX.sym('ds')
        state = ca.vertcat(x,y,v,psi,delta,ey,epsi,t)
        
        # ODE
        x_prime = ((1 - ey*kappa) * cos(psi)) / (cos(epsi))
        y_prime = ((1 - ey*kappa) * sin(psi)) / (cos(epsi))
        psi_prime = ((1 - ey*kappa) * tan(delta)) / (self.car.length * cos(epsi))
        v_prime = (1 - ey * kappa) / (v * np.cos(epsi)) * a
        delta_prime = (1 - ey * kappa) / (v * np.cos(epsi)) * w
        ey_prime = (1 - ey * kappa) * ca.tan(epsi)
        epsi_prime = ((tan(delta)) / self.car.length) * ((1 - ey * kappa)/(np.cos(epsi))) - kappa
        t_prime = (1 - ey * kappa) / (v * np.cos(epsi))
        state_prime = ca.vertcat(x_prime, y_prime, v_prime, psi_prime, delta_prime, ey_prime, epsi_prime, t_prime)
        ode = ca.Function('ode', [state, input, kappa], [state_prime])
        
        # wrapping up
        integrator = integrate(state, input, kappa, ode, h=ds) # TODO ds
        transition = ca.Function('transition', [state,input,kappa,ds], [integrator])
        
        return ns, ni, transition
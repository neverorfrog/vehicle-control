from omegaconf import OmegaConf
from models.dynamic_car import DynamicCar, DynamicCarInput
import casadi as ca
import numpy as np
from casadi import cos, tan, fabs
from controllers.controller import Controller
from models.kinematic_car import KinematicCar
np.random.seed(31)

class CascadedKinematicMPC(Controller):
    def __init__(self, car: DynamicCar, kin_car: KinematicCar, config: OmegaConf):
        """Optimizer Initialization"""
        self.config = config
        self.car = car
        self.kin_car = kin_car
        self._init_dims()
        self._init_opti()
        self._init_variables()
        cost = 0
        self.opti.subject_to(self.state[:,0] == self.state0) # constraint on initial state
        for n in range(self.N):
            self._stage_constraints(n) 
            cost += self._stage_cost(n)
        if self.N > 0 and self.K > 0:
            cost += self._switching_cost()
            self._switching_constraints()
        for k in range(self.N,self.H):
            self._kc_stage_constraints(k)
            cost += self._kc_stage_cost(k)
        cost += self._terminal_cost()
        self.opti.minimize(cost)
            
    def _init_dims(self):
        # single-track
        self.N = self.config.horizon
        self.dt = self.config.mpc_dt
        self.ns = len(self.car.state) # number of state variables
        self.na = len(self.car.input) # number of action variables
        # point-mass
        self.K = self.config.horizon_kc
        self.dt_kc = self.config.mpc_dt_kc
        self.ns_kc = len(self.kin_car.state) # number of state variables
        self.na_kc = len(self.kin_car.input) # number of action variables
        self.H = self.N + self.K
        
    def _init_opti(self):
        self.opti = ca.Opti('nlp')
        ipopt_options = {
            'print_level': 2, 
            'linear_solver': 'ma27', 
            'hsllib': '/usr/local/lib/libcoinhsl.so',
            # 'fixed_variable_treatment': 'relax_bounds',
            'warm_start_init_point': 'yes',
            # 'warm_start_bound_push': 1e-8,}
            'nlp_scaling_method': 'gradient-based',
            'nlp_scaling_max_gradient': 100}
        options = {
            'print_time': False, 
            'expand': True, 
            'ipopt': ipopt_options}
        self.opti.solver("ipopt", options)
        
    def _init_variables(self):
        #state and action variables
        self.state = self.opti.variable(self.ns, self.H)
        self.action = self.opti.variable(self.na, self.H)
        self.state_prediction = np.ones((self.ns, self.H))
        self.state_prediction[self.car.state.index('Ux'),:self.N] += 3
        self.action_prediction = np.ones((self.na, self.H)) + np.random.random((self.na, self.H))        
        
        # track related stuff
        self.curvature = self.opti.parameter(self.H)
        self.ds = self.opti.parameter(self.H) # ds trajectory var (just for loggin purposes)
        
        # initial state
        self.state0 = self.opti.parameter(self.ns) if self.N > 0 else self.opti.parameter(self.ns_kc) 
    
    def _stage_constraints(self, n):
        state = self.state[:self.ns,n]; action = self.action[:,n]
        Ux,Uy,r,delta,s,ey,epsi,t = self._unpack_state(state)
        Fx,w = self._unpack_action(action)
        state_constraints = self.config.state_constraints
        input_constraints = self.config.input_constraints
        Peng = self.car.config.car.Peng
        mu = self.car.config.env.mu
        
        # state limits
        self.opti.subject_to(Ux >= state_constraints.Ux_min)
        self.opti.subject_to(self.opti.bounded(state_constraints.delta_min,delta,state_constraints.delta_max))
        
        # input limits
        self.opti.subject_to(Fx <= Peng / Ux)
        self.opti.subject_to(self.opti.bounded(input_constraints.w_min,w,input_constraints.w_max))
            
        # Model dynamics 
        if n < self.N - 1:
            self.opti.subject_to(self.state[:,n+1] == self.car.spatial_transition(state,action,self.curvature[n],self.ds[n]))        
        
        # longitudinal force limits on tires
        bound_f = mu.f*self.car.Fz_f(Ux,Fx)*cos(self.car.alpha_f(Ux,Uy,r,delta))
        self.opti.subject_to(self.opti.bounded(-bound_f,self.car.Fx_f(Fx),bound_f))
        bound_r = mu.r*self.car.Fz_r(Ux,Fx)*cos(self.car.alpha_r(Ux,Uy,r,delta))
        self.opti.subject_to(self.opti.bounded(-bound_r,self.car.Fx_r(Fx),bound_r))
        
           
    def _stage_cost(self, n):
        Ux,Uy,r,delta,s,ey,epsi,t = self._unpack_state(self.state[:self.ns,n])
        Fx,w = self._unpack_action(self.action[:,n])
        ds = self.ds[n]
        cost_weights = self.config.cost_weights
        state_constraints = self.config.state_constraints

        cost = 0
        
        cost += ca.if_else(ey < state_constraints.ey_min, # violation of road bounds
                       cost_weights.boundary*ds*(ey - state_constraints.ey_min)**2, 0)
            
        cost += ca.if_else(ey > state_constraints.ey_max, # violation of road bounds
                       cost_weights.boundary*ds*(ey - state_constraints.ey_max)**2, 0)
        
        cost += cost_weights.deviation_st*ds*(ey**2) # deviation from road desciptor
        
        cost += cost_weights.w*(w**2) # steer angle rate
        
        cost += ca.if_else(fabs(tan(self.car.alpha_f(Ux,Uy,r,delta))) >= tan(self.car.alphamod_f(Fx)),  # slip angle front
                        cost_weights.slip*(fabs(tan(self.car.alpha_f(Ux,Uy,r,delta))) - tan(self.car.alphamod_f(Fx)))**2, 0)
                        
        cost += ca.if_else(fabs(tan(self.car.alpha_r(Ux,Uy,r,delta))) >= tan(self.car.alphamod_r(Fx)),  # slip angle rear
                    cost_weights.slip*(fabs(tan(self.car.alpha_r(Ux,Uy,r,delta))) - tan(self.car.alphamod_r(Fx)))**2, 0)
        
        if n < self.N-1: #Force Input Continuity
            next_action = self.action[:,n+1]
            cost += (cost_weights.Fx/ds) * (next_action[self.car.input.index('Fx')] - Fx)**2
            
        if self.config.obstacles: #Obstacle avoidance
            for obs in self.car.track.obstacles:
                distance = ca.fabs(ca.sqrt((s - obs.s)**2 + (ey - obs.ey)**2) - (obs.radius))
                cost += ds/((n+1)*distance)
                
        return cost
    
    
    def _kc_stage_constraints(self, k):
        state = self.state[:self.ns_kc,k]; action = self.action[:,k]
        v,delta,s,ey,epsi,t = self._unpack_kc_state(state)
        a, w = self._unpack_kc_action(action)
        state_kc_constraints = self.config.state_kc_constraints
        input_constraints = self.config.input_constraints
        Peng = self.car.config.car.Peng
        
        # state limits
        self.opti.subject_to(v >= state_kc_constraints.v_min)
        self.opti.subject_to(self.opti.bounded(state_kc_constraints.delta_min,delta,state_kc_constraints.delta_max))
        
        # input limits
        self.opti.subject_to(self.opti.bounded(input_constraints.a_min,a,input_constraints.a_max))
        self.opti.subject_to(self.opti.bounded(input_constraints.w_min,w,input_constraints.w_max))
        
        # Model dynamics
        if k < self.H-1:
            self.opti.subject_to(self.state[:self.ns_kc,k+1] == self.kin_car.spatial_transition(state,action,self.curvature[k],self.ds[k]))
    
    def _kc_stage_cost(self, k):
        cost = 0
        v,delta,s,ey,epsi,t = self._unpack_kc_state(self.state[:self.ns_kc,k])
        a, w = self._unpack_kc_action(self.action[:,k])
        ds = self.ds[k]
        cost_weights = self.config.cost_weights
        state_kc_constraints = self.config.state_kc_constraints
        
        cost += ca.if_else(ey < state_kc_constraints.ey_min, # 1) road boundary intrusion
                cost_weights.boundary*ds*(ey - state_kc_constraints.ey_min)**2, 0)
            
        cost += ca.if_else(ey > state_kc_constraints.ey_max, # 2) road boundary intrusion
                cost_weights.boundary*ds*(ey - state_kc_constraints.ey_max)**2, 0)
        
        cost += cost_weights.deviation_kc*ds*(ey**2) # 3) deviation from road descriptor path
        
        cost += cost_weights.w_kc*(w**2) # steer angle rate
        
        if k < self.K-1: # 4) Slew Rate
            a_next, _ = self._unpack_kc_action(self.action[:,k+1])
            cost += (cost_weights.a) * (a_next - a)**2
        
        if self.config.obstacles:
            for obs in self.car.track.obstacles:
                distance = ca.fabs(ca.sqrt((s - obs.s)**2 + (ey - obs.ey)**2) - (obs.radius))
                cost += ds/((k+1)*distance) #5) Obstacle Avoidance
        
        return cost  
    
    def _switching_cost(self):
        cost_weights = self.config.cost_weights
        Ux,Uy,r,delta,s,ey,epsi,t = self._unpack_state(self.state[:self.ns,self.N-1]) #final state
        Fx, w_final = self._unpack_action(self.action[:,self.N-2]) #final action
        Ux_dot = self.car.Ux_dot(Fx,Ux,Uy,r,delta)
        Uy_dot = self.car.Uy_dot(Fx,Ux,Uy,r,delta)
        a_final = (Ux_dot**2 + Uy_dot**2)**0.5
        v_initial = (Ux**2 + Uy**2)**0.5
        Fy_initial = (v_initial**2 * delta) / self.car.config.car.l * self.car.config.car.m
        a_initial, w_initial = self._unpack_kc_action(self.action[:,self.N])
        return (cost_weights.Fy/self.ds[self.N]) * ((a_final-a_initial)**2 + (w_final-w_initial)**2)
        
    
    def _switching_constraints(self):
        Ux_final,Uy_final,r_final,delta_final,s_final,ey_final,epsi_final,t_final = self._unpack_state(self.state[:,self.N-1]) #final state
        state_kc_initial = self.state[:,self.N] # initial state of point mass
        self.opti.subject_to(state_kc_initial[self.kin_car.state.index('v')] == (Ux_final**2 + Uy_final**2)**0.5)
        self.opti.subject_to(state_kc_initial[self.kin_car.state.index('ey')] == ey_final)
        self.opti.subject_to(state_kc_initial[self.kin_car.state.index('epsi')] == ca.atan(Uy_final/Ux_final) + epsi_final)
        self.opti.subject_to(state_kc_initial[self.kin_car.state.index('s')] == s_final)
        self.opti.subject_to(state_kc_initial[self.kin_car.state.index('t')] == t_final)
        self.opti.subject_to(state_kc_initial[self.kin_car.state.index('delta')] == delta_final)

        
    def _terminal_cost(self):
        cost = 0
        state_constraints = self.config.state_constraints
        cost_weights = self.config.cost_weights
        final_state = self.state[:,-1]
        if self.K > 0:
            final_model = self.kin_car
            final_speed = final_state[final_model.state.index('v')]
        else:
            final_model = self.car
            final_speed = final_state[final_model.state.index('Ux')]
        cost += ca.if_else(final_speed >= state_constraints.max_speed,cost_weights.speed*(final_speed - state_constraints.max_speed)**2, 0) # excessive speed
        cost += cost_weights.time*(final_state[final_model.state.index('t'),-1]) # final cost (minimize time)
        cost += cost_weights.ey*final_state[final_model.state.index('ey'),-1]**2 # final cost (minimize terminal lateral error) hardcodato
        cost += cost_weights.epsi*final_state[final_model.state.index('epsi'),-1]**2 # final cost (minimize terminal course error) hardcodato
        return cost
    
    
    def command(self, state):
        self._init_horizon(state)
        sol = self.opti.solve()
        self.action_prediction = sol.value(self.action)
        self.state_prediction = sol.value(self.state)
        return DynamicCarInput(Fx=self.action_prediction[0][0], w=self.action_prediction[1][0]), sol
    
    def _init_horizon(self, state):
        #initial state
        self.opti.set_value(self.state0, state.values.squeeze())
        #initializing state and action prediction
        self.opti.set_initial(self.action, self.action_prediction)
        self.opti.set_initial(self.state, self.state_prediction)
        #initializing car s trajectory
        ds_traj = np.full(self.N, self.config.mpc_dt) * self.state_prediction[self.car.state.index('Ux'),:self.N]
        self.opti.set_value(self.ds[:self.N], ds_traj)
        #initializing car k trajectory
        s_traj = np.cumsum(ds_traj) - ds_traj[0] + state[self.car.state.index('s')]
        self.opti.set_value(self.curvature[:self.N], self.car.track.k(s_traj))
        #initializing kc s trajectory
        ds_kc_traj = np.full(self.K, self.config.mpc_dt_kc) * self.state_prediction[self.kin_car.state.index('v'),self.N:self.H]
        self.opti.set_value(self.ds[self.N:self.H], ds_kc_traj)
        #initializing kc k trajectory
        s_kc_traj = np.cumsum(ds_kc_traj) - ds_traj[-1] + s_traj[-1]
        self.opti.set_value(self.curvature[self.N:self.H], self.car.track.k(s_kc_traj))
        
    def get_state_prediction(self):
        preds_car = [self.car.rel2glob(self.state_prediction[:,i]) for i in range(self.N)]
        preds_kin = [self.kin_car.rel2glob(self.state_prediction[:,k]) for k in range(self.N,self.H)] if self.K > 0 else []
        return np.array(preds_car + preds_kin).squeeze()
            
    def _unpack_state(self, state):
        Ux = state[self.car.state.index('Ux')]
        Uy = state[self.car.state.index('Uy')]
        r = state[self.car.state.index('r')]
        delta = state[self.car.state.index('delta')]
        s = state[self.car.state.index('s')]
        ey = state[self.car.state.index('ey')]
        epsi = state[self.car.state.index('epsi')]
        t = state[self.car.state.index('t')]
        return Ux,Uy,r,delta,s,ey,epsi,t
    
    def _unpack_action(self, action):
        Fx = action[self.car.input.index('Fx')]
        w = action[self.car.input.index('w')]
        return Fx,w

    def _unpack_kc_state(self, state):
        v = state[self.kin_car.state.index('v')]
        s = state[self.kin_car.state.index('s')]
        ey = state[self.kin_car.state.index('ey')]
        epsi = state[self.kin_car.state.index('epsi')]
        t = state[self.kin_car.state.index('t')]
        delta = state[self.kin_car.state.index('delta')]
        return v,delta,s,ey,epsi,t
    
    def _unpack_kc_action(self, action):
        a = action[self.kin_car.input.index('a')]
        w = action[self.kin_car.input.index('w')]
        return a,w
        
    
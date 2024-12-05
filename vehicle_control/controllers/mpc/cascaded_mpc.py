from typing import Tuple
from omegaconf import OmegaConf
from vehicle_control.controllers.mpc.kinematic_mpc import KinematicMPC
from vehicle_control.models.dynamic_car import DynamicCar, DynamicCarAction
from vehicle_control.models.dynamic_point_mass import DynamicPointMass
import casadi as ca
import numpy as np
from casadi import cos, tan, fabs
from vehicle_control.controllers.controller import Controller
from vehicle_control.utils.fancy_vector import FancyVector
np.random.seed(31)

class CascadedMPC(Controller):
    def __init__(self, car: DynamicCar, point_mass: DynamicPointMass, config: OmegaConf):
        """Optimizer Initialization"""
        self.config = config
        self.car = car
        self.point_mass = point_mass
        self._init_dims()
        self._init_opti()
        self._init_variables()
        cost = 0
        self.opti.subject_to(self.state[:,0] == self.state0) # constraint on initial state
        for n in range(self.N):
            self._stage_constraints(n) 
            cost += self._stage_cost(n)
        if self.N > 0 and self.M > 0:
            cost += self._switching_cost()
            self._switching_constraints()
        for m in range(self.N,self.H):
            self._pm_stage_constraints(m)
            cost += self._pm_stage_cost(m)
        cost += self._terminal_cost()
        self.opti.minimize(cost)
            
    def _init_dims(self):
        # single-track
        self.N = self.config.horizon
        self.dt = self.config.mpc_dt
        self.ns = len(self.car.state) # number of state variables
        self.na = len(self.car.input) # number of action variables
        # point-mass
        self.M = self.config.horizon_pm
        self.ns_pm = len(self.point_mass.state) # number of state variables
        self.na_pm = len(self.point_mass.input) # number of action variables
        self.H = self.N + self.M
            
    def _init_opti(self):
        self.opti = ca.Opti('nlp')
        ipopt_options = {
            'print_level': 2, 
            'hsllib': '/usr/local/lib/libcoinhsl.so',
            'warm_start_init_point': 'yes',
            # 'fixed_variable_treatment': 'relax_bounds',
            # 'warm_start_bound_push': 1e-8,
            # 'nlp_scaling_method': 'gradient-based',
            # 'nlp_scaling_max_gradient': 100,
            'linear_solver': 'ma27'}
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
        self.state0 = self.opti.parameter(self.ns) if self.N > 0 else self.opti.parameter(self.ns_pm) 
        
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
        
        talpha_f = fabs(tan(self.car.alpha_f(Ux,Uy,r,delta)))
        talphamod_f = tan(self.car.alphamod_f(Fx))
        cost += ca.if_else(talpha_f >= talphamod_f, cost_weights.slip*(talpha_f - talphamod_f)**2, 0) #slip angle front
                        
        talpha_r = fabs(tan(self.car.alpha_r(Ux,Uy,r,delta)))
        talphamod_r = tan(self.car.alphamod_r(Fx))
        cost += ca.if_else(talpha_r >= talphamod_r, cost_weights.slip*(talpha_r - talphamod_r)**2, 0) #slip angle rear
        
        if n < self.N-1: #Force Action Continuity
            next_action = self.action[:,n+1]
            cost += (cost_weights.Fx/ds) * (next_action[self.car.input.index('Fx')] - Fx)**2
            
        if self.config.obstacles: #Obstacle avoidance
            for obs in self.car.track.obstacles:
                distance = ca.sqrt((s - obs.s)**2 + (ey - obs.ey)**2)
                cost += cost_weights.obstacles*ds/(distance-(obs.radius+0.1))
                # cost += ca.if_else(distance <= obs.radius + 0.1, cost_weights.obstacles*ds/(distance-obs.radius), 0) 
                
        return cost
    
    
    def _pm_stage_constraints(self, m):
        state = self.state[:self.ns_pm,m]; action = self.action[:,m]
        V,s,ey,epsi,t = self._unpack_pm_state(state)
        Fx, Fy = self._unpack_pm_action(action)
        state_pm_constraints = self.config.state_pm_constraints
        Peng = self.car.config.car.Peng
        
        # state limits
        self.opti.subject_to(V >= state_pm_constraints.V_min)
        
        # input limits
        self.opti.subject_to(Fx <= Peng / V)
        
        # Model dynamics
        if m < self.H-1:
            self.opti.subject_to(self.state[:self.ns_pm,m+1] == self.point_mass.spatial_transition(state,action,self.curvature[m],self.ds[m]))
        
    
    def _pm_stage_cost(self, m):
        cost = 0
        V,s,ey,epsi,t = self._unpack_pm_state(self.state[:self.ns_pm,m])
        Fx, Fy = self._unpack_pm_action(self.action[:,m])
        ds = self.ds[m]
        cost_weights = self.config.cost_weights
        state_pm_constraints = self.config.state_pm_constraints
        
        cost += ca.if_else(ey < state_pm_constraints.ey_min, # 1) road boundary intrusion
                cost_weights.boundary*ds*(ey - state_pm_constraints.ey_min)**2, 0)
            
        cost += ca.if_else(ey > state_pm_constraints.ey_max, # 2) road boundary intrusion
                cost_weights.boundary*ds*(ey - state_pm_constraints.ey_max)**2, 0)
        
        cost += cost_weights.deviation_pm*ds*(ey**2) # 3) deviation from road descriptor path
        
        if m < self.H-1: # 4) Slew Rate
            Fx_next, Fy_next = self._unpack_pm_action(self.action[:,m+1])
            cost += cost_weights.Fy*(1/ds)*(Fy_next - Fy)**2
            cost += cost_weights.Fx*(1/ds)*(Fx_next - Fx)**2 
        
        if self.config.obstacles:
            for obs in self.car.track.obstacles:
                distance = ca.sqrt((s - obs.s)**2 + (ey - obs.ey)**2)
                cost += cost_weights.obstacles*ds/(distance-(obs.radius+0.1))
                # cost += ca.if_else(distance <= obs.radius + 0.1, cost_weights.obstacles_pm*ds/(distance-obs.radius), 0) #5) Obstacle Avoidance
        
        return cost  
    
    def _switching_cost(self):
        cost_weights = self.config.cost_weights
        Ux_final,Uy_final,r_final,delta_final,_,_,_,_ = self._unpack_state(self.state[:self.ns,self.N-1]) #final state
        Fx_final, _ = self._unpack_action(self.action[:,self.N-1]) #final action
        Fx_bar_initial, Fy_bar_initial = self._unpack_pm_action(self.action[:,self.N])
        Fy_f = self.car.Fy_f(Ux_final,Uy_final,r_final,delta_final,Fx_final)
        Fy_r = self.car.Fy_r(Ux_final,Uy_final,r_final,delta_final,Fx_final)
        return cost_weights.switch_F * (1/self.ds[self.N-1]) * (((Fx_bar_initial-Fx_final)**2) + (Fy_bar_initial-Fy_f-Fy_r)**2)
        
    
    def _switching_constraints(self):
        Ux_final,Uy_final,_,_,s_final,ey_final,epsi_final,t_final = self._unpack_state(self.state[:,self.N-1]) #final state
        state_pm_initial = self.state[:,self.N] # initial state of point mass
        self.opti.subject_to(state_pm_initial[self.point_mass.state.index('V')] == (Ux_final**2 + Uy_final**2)**0.5)
        self.opti.subject_to(state_pm_initial[self.point_mass.state.index('ey')] == ey_final)
        self.opti.subject_to(state_pm_initial[self.point_mass.state.index('epsi')] == ca.atan(Uy_final/Ux_final) + epsi_final)
        self.opti.subject_to(state_pm_initial[self.point_mass.state.index('s')] == s_final)
        self.opti.subject_to(state_pm_initial[self.point_mass.state.index('t')] == t_final)

        
    def _terminal_cost(self):
        cost = 0
        state_constraints = self.config.state_constraints
        cost_weights = self.config.cost_weights
        final_state = self.state[:,-1]
        if self.M > 0:
            final_model = self.point_mass
            final_speed = final_state[final_model.state.index('V')]
        else:
            final_model = self.car
            final_speed = final_state[final_model.state.index('Ux')]
        cost += ca.if_else(final_speed >= state_constraints.max_speed,cost_weights.speed*(final_speed - state_constraints.max_speed)**2, 0) # excessive speed
        cost += cost_weights.time*(final_state[final_model.state.index('t')]) # final cost (minimize time)
        cost += cost_weights.ey*final_state[final_model.state.index('ey')]**2 # final cost (minimize terminal lateral error) hardcodato
        cost += cost_weights.epsi*final_state[final_model.state.index('epsi')]**2 # final cost (minimize terminal course error) hardcodato
        return cost
    
    
    def command(self, state) -> Tuple[FancyVector, FancyVector]:
        self._init_horizon(state)
        sol = self.opti.solve()
        self.action_prediction = sol.value(self.action)
        self.state_prediction = sol.value(self.state)
        action = DynamicCarAction(Fx=self.action_prediction[0][0], w=self.action_prediction[1][0])
        return action

    
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
        #initializing pm s trajectory
        ds_pm_traj = np.full(self.M, self.config.ds_pm) #self.config.mpc_dt_pm) * self.state_prediction[self.point_mass.state.index('V'),self.N:self.H]
        self.opti.set_value(self.ds[self.N:self.H], ds_pm_traj)
        #initializing pm k trajectory
        s_pm_traj = np.cumsum(ds_pm_traj) - ds_traj[-1] + s_traj[-1]
        self.opti.set_value(self.curvature[self.N:self.H], self.car.track.k(s_pm_traj))
        
    def get_state_prediction(self):
        preds_car = [self.car.rel2glob(self.state_prediction[:,i]) for i in range(self.N)]
        preds_pm = [self.point_mass.rel2glob(self.state_prediction[:,j]) for j in range(self.N,self.H)] if self.M > 0 else []
        return np.array(preds_car + preds_pm).squeeze()
            
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

    def _unpack_pm_state(self, state):
        V = state[self.point_mass.state.index('V')]
        s = state[self.point_mass.state.index('s')]
        ey = state[self.point_mass.state.index('ey')]
        epsi = state[self.point_mass.state.index('epsi')]
        t = state[self.point_mass.state.index('t')]
        return V,s,ey,epsi,t
    
    def _unpack_pm_action(self, action):
        Fx = action[self.point_mass.input.index('Fx')]
        Fy = action[self.point_mass.input.index('Fy')]
        return Fx,Fy
        
    
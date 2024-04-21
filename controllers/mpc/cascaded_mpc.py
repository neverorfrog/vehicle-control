from omegaconf import OmegaConf
from models.dynamic_car import DynamicCar, DynamicCarInput
from models.dynamic_point_mass import DynamicPointMass
import casadi as ca
import numpy as np
from casadi import cos, tan, fabs
from controllers.controller import Controller
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
        if self.M > 0:
            cost += self._switching_cost()
            self._switching_constraints()
        for m in range(self.M):
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
        self.dt_pm = self.config.mpc_dt_pm
        self.ns_pm = len(self.point_mass.state) # number of state variables
        self.na_pm = len(self.point_mass.input) # number of action variables
            
    def _init_opti(self):
        self.opti = ca.Opti('nlp')
        ipopt_options = {'print_level': 1, 'linear_solver': 'ma27', 'hsllib': '/usr/local/lib/libcoinhsl.so', 'fixed_variable_treatment': 'relax_bounds'}
        options = {'print_time': False, 'expand': True, 'ipopt': ipopt_options}
        self.opti.solver("ipopt", options)
    
    def _init_variables(self):
        # initial state
        self.state0 = self.opti.parameter(self.ns)
        
        # single-track
        self.state = self.opti.variable(self.ns, self.N+1) # state trajectory var
        self.action = self.opti.variable(self.na, self.N)  # control trajectory var
        self.ds = self.opti.parameter(self.N) # ds trajectory var (just for loggin purposes)
        self.state_prediction = np.zeros((self.ns, self.N+1)); 
        self.state_prediction[self.car.state.index('Ux'),:] += 4
        self.state_prediction[self.car.state.index('s'),:] += 1
        self.action_prediction = np.ones((self.na, self.N)) + np.random.random((self.na, self.N))
        self.curvature = self.opti.parameter(self.N) # curvature trajectory
        
        # point-mass
        self.state_pm = self.opti.variable(self.ns_pm, self.M+1)
        self.action_pm = self.opti.variable(self.na_pm, self.M)
        self.ds_pm = self.opti.parameter(self.M) # ds trajectory var (just for loggin purposes)
        self.state_pm_prediction = np.ones((self.ns_pm, self.M+1))
        self.action_pm_prediction = np.ones((self.na_pm, self.M)) + np.random.random((self.na_pm, self.M))
        self.curvature_pm = self.opti.parameter(self.M)
        
        # slack variables
        self.Fe_f = self.opti.variable(self.N) 
        self.Fe_r = self.opti.variable(self.N)
    
    def _stage_constraints(self, n):
        state = self.state[:,n]; action = self.action[:,n]
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
        self.opti.subject_to(self.state[:,n+1] == self.car.spatial_transition(state,action,self.curvature[n],self.ds[n]))
        
        # longitudinal force limits on tires
        bound_f = mu.f*self.car.Fz_f(Ux,Fx)*cos(self.car.alpha_f(Ux,Uy,r,delta))
        self.opti.subject_to(self.opti.bounded(-bound_f,self.car.Fx_f(Fx),bound_f))
        bound_r = mu.r*self.car.Fz_r(Ux,Fx)*cos(self.car.alpha_r(Ux,Uy,r,delta))
        self.opti.subject_to(self.opti.bounded(-bound_r,self.car.Fx_r(Fx),bound_r))
        
           
    def _stage_cost(self, n):
        Ux,Uy,r,delta,s,ey,epsi,t = self._unpack_state(self.state[:,n])
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
        
        cost += cost_weights.friction*((self.Fe_f[n]**2)**2 + (self.Fe_r[n]**2)**2) # slack variables for sparsity
        
        if n < self.N-1: #Force Input Continuity
            next_action = self.action[:,n+1]
            cost += (cost_weights.Fx/ds) * (next_action[self.car.input.index('Fx')] - Fx)**2
            
        if self.config.obstacles: #Obstacle avoidance
            for obs in self.car.track.obstacles:
                distance = ca.fabs(ca.sqrt((s - obs.s)**2 + (ey - obs.ey)**2) - (obs.radius))
                cost += 10*(1/distance)
                
        return cost
    
    
    def _pm_stage_constraints(self, m):
        V,s,ey,epsi,t = self._unpack_pm_state(self.state_pm[:,m])
        Fx, Fy = self._unpack_pm_action(self.action_pm[:,m])
        state_pm_constraints = self.config.state_pm_constraints
        input_constraints = self.config.input_constraints
        Peng = self.car.config.car.Peng
        
        # state limits
        self.opti.subject_to(V >= state_pm_constraints.V_bar_min)
        
        # input limits
        self.opti.subject_to(Fx <= Peng / V)
        
        # Model dynamics
        self.opti.subject_to(self.state_pm[:,m+1] == self.point_mass.spatial_transition(self.state_pm[:,m],self.action_pm[:,m],self.curvature_pm[m],self.ds_pm[m]))
        
        # friction limits
        Fx_f_bar = self.car.Fx_f(Fx)
        Fx_r_bar = self.car.Fx_r(Fx)
        self.opti.subject_to(Fx_f_bar**2 + (self.car.config.car.b/self.car.config.car.l*Fy)**2 <= (input_constraints.mu_lim*self.car.Fz_f(V,Fx))**2)
        self.opti.subject_to(Fx_r_bar**2 + (self.car.config.car.a/self.car.config.car.l*Fy)**2 <= (input_constraints.mu_lim*self.car.Fz_r(V,Fx))**2)
        
    
    def _pm_stage_cost(self, m):
        cost = 0
        V,s,ey,epsi,t = self._unpack_pm_state(self.state_pm[:,m])
        Fx, Fy = self._unpack_pm_action(self.action_pm[:,m])
        ds = self.ds_pm[m]
        cost_weights = self.config.cost_weights
        state_pm_constraints = self.config.state_pm_constraints
        
        cost += ca.if_else(ey < state_pm_constraints.ey_bar_min, # 1) road boundary intrusion
                cost_weights.boundary*ds*(ey - state_pm_constraints.ey_bar_min)**2, 0)
            
        cost += ca.if_else(ey > state_pm_constraints.ey_bar_max, # 2) road boundary intrusion
                cost_weights.boundary*ds*(ey - state_pm_constraints.ey_bar_max)**2, 0)
        
        cost += cost_weights.deviation_pm*ds*(ey**2) # 3) deviation from road descriptor path
        
        if m < self.M-1: # 4) Slew Rate
            Fx_next, Fy_next = self._unpack_pm_action(self.action_pm[:,m+1])
            cost += cost_weights.Fy*(1/ds)*(Fy_next - Fy)**2
            cost += cost_weights.Fx*(1/ds)*(Fx_next - Fx)**2 
        
        if self.config.obstacles:
            for obs in self.car.track.obstacles:
                distance = ca.fabs(ca.sqrt((s - obs.s)**2 + (ey - obs.ey)**2) - (obs.radius))
                cost += 10*(1/distance) #5) Obstacle Avoidance
        
        return cost  
    
    def _switching_cost(self):
        cost_weights = self.config.cost_weights
        Ux_final,Uy_final,r_final,delta_final,s_final,ey_final,epsi_final,t_final = self._unpack_state(self.state[:,-1]) #final state
        Fx_final, w_final = self._unpack_action(self.action[:,-1]) #final action
        Fx_bar_initial, Fy_bar_initial = self._unpack_pm_action(self.action_pm[:,0])
        Fy_f = self.car.Fy_f(Ux_final,Uy_final,r_final,delta_final,Fx_final)
        Fy_r = self.car.Fy_r(Ux_final,Uy_final,r_final,delta_final,Fx_final)
        return (cost_weights.Fx/self.ds[-1]) * (((Fx_bar_initial-Fx_final)**2)+ (Fy_bar_initial-Fy_f-Fy_r)**2)
        
    
    def _switching_constraints(self):
        Ux_final,Uy_final,r_final,delta_final,s_final,ey_final,epsi_final,t_final = self._unpack_state(self.state[:,-1]) #final state
        state_pm_initial = self.state_pm[:,0] # initial state of point mass
        self.opti.subject_to(state_pm_initial[self.point_mass.state.index('V')] == (Ux_final**2 + Uy_final**2)**0.5)
        self.opti.subject_to(state_pm_initial[self.point_mass.state.index('ey')] == ey_final)
        self.opti.subject_to(state_pm_initial[self.point_mass.state.index('epsi')] == ca.atan(Uy_final/Ux_final) + epsi_final)
        self.opti.subject_to(state_pm_initial[self.point_mass.state.index('s')] == s_final)
        self.opti.subject_to(state_pm_initial[self.point_mass.state.index('t')] == t_final)

        
    def _terminal_cost(self):
        cost = 0
        state_constraints = self.config.state_constraints
        cost_weights = self.config.cost_weights
        if self.M > 0:
            final_state = self.state_pm[:,-1]
            final_model = self.point_mass
            final_speed = final_state[final_model.state.index('V')]
        else:
            final_state = self.state[:,-1]
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
        self._save_horizon(sol)
        return DynamicCarInput(Fx=self.action_prediction[0][0], w=self.action_prediction[1][0]), sol
    
    def _init_horizon(self, state):
        # initial state
        state = state.values.squeeze()
        self.opti.set_value(self.state0, state)
        
        # initializing state and action prediction
        self.opti.set_initial(self.action, self.action_prediction)
        self.opti.set_initial(self.state, self.state_prediction)
        #initializing s and k trajectory
        ds_traj = np.full(self.N+1, self.config.mpc_dt) * self.state_prediction[self.car.state.index('Ux'),:]
        self.opti.set_value(self.ds, ds_traj[:-1])
        ds_traj[0] = 0; s_traj = np.cumsum(ds_traj) + state[self.car.state.index('s')]; s_traj = s_traj[:-1]
        self.opti.set_value(self.curvature, self.car.track.k(s_traj))
        
        # same for point-mass
        if self.M > 0:
            self.opti.set_initial(self.action_pm, self.action_pm_prediction)
            self.opti.set_initial(self.state_pm, self.state_pm_prediction)
            #initializing s and k trajectory
            ds_bar_traj = np.full(self.M+1, self.config.mpc_dt_pm) * self.state_pm_prediction[self.point_mass.state.index('V'),:]
            # ds_bar_traj = np.full(self.M+1, self.config.ds_bar)
            self.opti.set_value(self.ds_pm, ds_bar_traj[:-1])
            ds_bar_traj[0] = 0; s_bar_traj = np.cumsum(ds_bar_traj) + s_traj[-1]
            self.opti.set_value(self.curvature_pm, self.car.track.k(s_bar_traj[:-1]))
            
    def _save_horizon(self, sol):
        # saving for warmstart
        self.action_prediction = sol.value(self.action)
        self.state_prediction = sol.value(self.state)
        if self.M > 0:
            self.action_pm_prediction = sol.value(self.action_pm)
            self.state_pm_prediction = sol.value(self.state_pm)
            
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
        
    
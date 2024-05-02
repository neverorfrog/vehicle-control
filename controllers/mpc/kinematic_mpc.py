from omegaconf import OmegaConf
from models.kinematic_car import KinematicCar, KinematicCarAction
import casadi as ca
import numpy as np
from controllers.controller import Controller
np.random.seed(31)

class KinematicMPC(Controller):
    def __init__(self, car: KinematicCar, config: OmegaConf):
        """Optimizer Initialization"""
        self.config = config
        self.car = car
        self._init_dims()
        self._init_opti()
        self._init_variables()
        cost = 0
        self.opti.subject_to(self.state[:,0] == self.state0) # constraint on initial state
        for n in range(self.N):
            self._stage_constraints(n) 
            cost += self._stage_cost(n)
        cost += self._terminal_cost()
        self.opti.minimize(cost)
            
    def _init_dims(self):
        # single-track
        self.N = self.config.horizon
        self.dt = self.config.mpc_dt
        self.ns = len(self.car.state) # number of state variables
        self.na = len(self.car.input) # number of action variables
            
    def _init_opti(self):
        self.opti = ca.Opti('nlp')
        ipopt_options = {
            'print_level': 2, 
            'linear_solver': 'ma27', 
            'hsllib': '/usr/local/lib/libcoinhsl.so',
            # 'fixed_variable_treatment': 'relax_bounds',
            'warm_start_init_point': 'yes',
            'warm_start_bound_push': 1e-8,
            'nlp_scaling_method': 'gradient-based',
            'nlp_scaling_max_gradient': 100}
        options = {
            'print_time': False, 
            'expand': True, 
            'ipopt': ipopt_options}
        self.opti.solver("ipopt", options)
    
    def _init_variables(self):
        # initial state
        self.state0 = self.opti.parameter(self.ns)
        
        # single-track
        self.state = self.opti.variable(self.ns, self.N+1) # state trajectory var
        self.action = self.opti.variable(self.na, self.N)  # control trajectory var
        self.ds = self.opti.parameter(self.N) # ds trajectory var (just for loggin purposes)
        self.state_prediction = np.zeros((self.ns, self.N+1)); 
        self.action_prediction = np.ones((self.na, self.N)) + np.random.random((self.na, self.N))
        self.state_prediction[self.car.state.index('v'),:] += 0.1
        self.curvature = self.opti.parameter(self.N) # curvature trajectory
    
    def _stage_constraints(self, n):
        state = self.state[:,n]; action = self.action[:,n]
        v,delta,s,ey,epsi,t = self._unpack_state(state)
        a,w = self._unpack_action(action)
        state_constraints = self.config.state_constraints
        input_constraints = self.config.input_constraints
        
        # state limits
        self.opti.subject_to(v >= state_constraints.v_min)
        self.opti.subject_to(self.opti.bounded(state_constraints.delta_min,delta,state_constraints.delta_max))
        
        # input limits
        self.opti.subject_to(self.opti.bounded(input_constraints.a_min,a,input_constraints.a_max))
        self.opti.subject_to(self.opti.bounded(input_constraints.w_min,w,input_constraints.w_max))
            
        # Model dynamics 
        self.opti.subject_to(self.state[:,n+1] == self.car.spatial_transition(state,action,self.curvature[n],self.ds[n]))
        
    def _stage_cost(self, n):
        v,delta,s,ey,epsi,t = self._unpack_state(self.state[:,n])
        a,w = self._unpack_action(self.action[:,n])
        ds = self.ds[n]
        cost_weights = self.config.cost_weights
        state_constraints = self.config.state_constraints

        cost = 0
        
        cost += ca.if_else(ey < state_constraints.ey_min, # violation of road bounds
                       cost_weights.boundary*ds*(ey - state_constraints.ey_min)**2, 0)
            
        cost += ca.if_else(ey > state_constraints.ey_max, # violation of road bounds
                       cost_weights.boundary*ds*(ey - state_constraints.ey_max)**2, 0)
        
        cost += cost_weights.deviation*ds*(ey**2) # deviation from road desciptor
        
        cost += cost_weights.w*(w**2) # steer angle rate
        
        if n < self.N-1: #Force Action Continuity
            next_action = self.action[:,n+1]
            cost += (cost_weights.a) * (next_action[self.car.input.index('a')] - a)**2
            
        if self.config.obstacles: #Obstacle avoidance
            for obs in self.car.track.obstacles:
                distance = ca.fabs(ca.sqrt((s - obs.s)**2 + (ey - obs.ey)**2) - (obs.radius))
                cost += 10*(1/distance)
                
        return cost
    
    def _terminal_cost(self):
        cost = 0
        state_constraints = self.config.state_constraints
        cost_weights = self.config.cost_weights
        final_state = self.state[:,-1]
        final_model = self.car
        final_speed = final_state[final_model.state.index('v')]
        cost += ca.if_else(final_speed >= state_constraints.v_max,cost_weights.v*(final_speed - state_constraints.v_max)**2, 0) # excessive speed
        cost += cost_weights.time*(final_state[final_model.state.index('t'),-1]) # final cost (minimize time)
        cost += cost_weights.ey*final_state[final_model.state.index('ey'),-1]**2 # final cost (minimize terminal lateral error) hardcodato
        cost += cost_weights.epsi*final_state[final_model.state.index('epsi'),-1]**2 # final cost (minimize terminal course error) hardcodato
        return cost
    
    
    def command(self, state):
        print("KINEMATIC BABY")
        self._init_horizon(state)
        sol = self.opti.solve()
        self.action_prediction = sol.value(self.action)
        self.state_prediction = sol.value(self.state)
        action = KinematicCarAction(a=self.action_prediction[0][0], w=self.action_prediction[1][0])
        state = self.car.drive(action)
        return action, state

    
    def _init_horizon(self, state):
        # initial state
        state = state.values.squeeze()
        self.opti.set_value(self.state0, state)
        # initializing state and action prediction
        self.opti.set_initial(self.action, self.action_prediction)
        self.opti.set_initial(self.state, self.state_prediction)
        #initializing s and k trajectory
        ds_traj = np.full(self.N+1, self.config.mpc_dt) * self.state_prediction[self.car.state.index('v'),:] + 0.5
        self.opti.set_value(self.ds, ds_traj[:-1])
        ds_traj[0] = 0; s_traj = np.cumsum(ds_traj) + state[self.car.state.index('s')]; s_traj = s_traj[:-1]
        self.opti.set_value(self.curvature, self.car.track.k(s_traj))

            
    def get_state_prediction(self):
        preds_car = [self.car.rel2glob(self.state_prediction[:,i]) for i in range(self.N)]
        return np.array(preds_car).squeeze()
            
    def _unpack_state(self, state):
        v = state[self.car.state.index('v')]
        delta = state[self.car.state.index('delta')]
        ey = state[self.car.state.index('ey')]
        epsi = state[self.car.state.index('epsi')]
        s = state[self.car.state.index('s')]
        t = state[self.car.state.index('t')]
        return v,delta,s,ey,epsi,t
    
    def _unpack_action(self, action):
        a = action[self.car.input.index('a')]
        w = action[self.car.input.index('w')]
        return a,w
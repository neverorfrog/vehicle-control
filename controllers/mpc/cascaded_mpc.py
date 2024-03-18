from models.dynamic_car import DynamicCar, DynamicCarInput
from models.dynamic_point_mass import DynamicPointMass
import casadi as ca
import numpy as np
from casadi import cos, sin, tan, fabs, atan2, atan
from controllers.controller import Controller
np.random.seed(3)

class CascadedMPC(Controller):
    def __init__(self, car: DynamicCar, point_mass: DynamicPointMass, config):
        self.dt = config['mpc_dt']
        self.dt_pm = config['mpc_dt_pm']
        self.car = car
        self.point_mass = point_mass
        self.N = config['horizon']
        self.M = config['horizon_pm']
        self.config = config
        self.ns = len(self.car.state) # number of state variables
        self.na = len(self.car.input) # number of action variables
        self.ns_pm = len(self.point_mass.state) # number of state variables
        self.na_pm = len(self.point_mass.input) # number of action variables
        self.opti = self._init_opti()
        
    def command(self, state):
        self._init_horizon(state)
        sol = self.opti.solve()
        self.action_prediction = sol.value(self.action)
        self.state_prediction = sol.value(self.state)
        print(f"Solver iterations: {sol.stats()["iter_count"]}")
        print(f"ds: {sol.value(self.ds)}")
        print(f"car_epsi: {self.state_prediction[self.car.state.index('epsi'),-1]}")
        print(f"car_ey: {self.state_prediction[self.car.state.index('ey'),-1]}")
        if self.M > 0:
            self.action_pm_prediction = sol.value(self.action_pm)
            self.state_pm_prediction = sol.value(self.state_pm)
            print(f"pm_ey: {self.state_pm_prediction[self.point_mass.state.index('ey'),0]}")
            print(f"pm_epsi: {self.state_pm_prediction[self.point_mass.state.index('epsi'),0]}")
            print(f"ds_pm: {sol.value(self.ds_pm)}")
        return DynamicCarInput(Fx=self.action_prediction[0][0], w=self.action_prediction[1][0])
    
    def _init_horizon(self, state):
        state = state.values.squeeze()
        self.opti.set_value(self.state0, state)
        self.opti.set_initial(self.action, self.action_prediction)
        self.opti.set_initial(self.state, self.state_prediction)
        if self.M > 0:
            self.opti.set_initial(self.action_pm, self.action_pm_prediction)
            self.opti.set_initial(self.state_pm, self.state_pm_prediction)
        
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
        self.action_prediction = np.zeros((self.na, self.N))   # actual predicted control trajectory
        self.ds = opti.variable(self.N) # ds trajectory var (just for loggin purposes)
        self.state0 = opti.parameter(self.ns) # initial state
        opti.subject_to(self.state[:,0] == self.state0) # constraint on initial state
        
        if self.M > 0:
            self.state_pm = opti.variable(self.ns_pm, self.M+1)
            self.action_pm = opti.variable(self.na_pm, self.M)
            self.state_pm_prediction = np.ones((self.ns_pm, self.M+1)) # actual predicted state trajectory
            self.action_pm_prediction = np.zeros((self.na_pm, self.M))   # actual predicted control trajectory
            self.ds_pm = opti.variable(self.M) # ds trajectory var (just for loggin purposes)

        
        # ======================== Slack Variables ============================================
        self.Fe_f = opti.variable(self.N) 
        self.Fe_r = opti.variable(self.N)  
        # self.Fy_f = opti.variable(self.N) 
        # self.Fy_r = opti.variable(self.N)
        # self.Fx_f = opti.variable(self.N) 
        # self.Fx_r = opti.variable(self.N) 
                                           
        # ======================= Cycle the entire horizon defining NLP problem ===============
        cost_weights = self.config['cost_weights'] 
        state_constraints = self.config['state_constraints']
        state_pm_constraints = self.config['state_pm_constraints']
        input_constraints = self.config['input_constraints']
        Peng = self.car.config['car']['Peng']
        mu = self.car.config['env']['mu']
        ds_bar = self.config['ds_bar']
        cost = 0
        
        # ============================ Single Track Model =================================
        for n in range(self.N):
            # extracting state and input at current iteration of horizon
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
            
            # -------------------- Stage Cost -------------------------------------
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
                cost += (cost_weights['Fx']/ds) * (Fx_next - Fx)**2 
                
            if n == self.N-1 and self.M > 0: #SINGLETRACK/POINTMASS Continuity
                Ux_final = state_next[self.car.state.index('Ux')]
                Uy_final = state_next[self.car.state.index('Uy')]
                ey_final = state_next[self.car.state.index('ey')]
                delta_final = state_next[self.car.state.index('delta')]
                s_final = state_next[self.car.state.index('s')]
                r_final = state_next[self.car.state.index('r')]
                t_final = state_next[self.car.state.index('t')]
                
                state_pm_initial = self.state_pm[:,0]
                opti.subject_to(state_pm_initial[self.point_mass.state.index('V')] == (Ux_final**2 + Uy_final**2)**0.5)
                opti.subject_to(state_pm_initial[self.point_mass.state.index('ey')] == ey_final)
                opti.subject_to(state_pm_initial[self.point_mass.state.index('epsi')] == atan(Uy_final/Ux_final) + delta_final)
                opti.subject_to(state_pm_initial[self.point_mass.state.index('s')] == s_final)
                opti.subject_to(state_pm_initial[self.point_mass.state.index('t')] == t_final)
                
                input_pm_initial = self.action_pm[:,0]
                Fx_bar_initial = input_pm_initial[self.point_mass.input.index('Fx')]
                Fy_bar_initial = input_pm_initial[self.point_mass.input.index('Fy')]
                Fy_f_final = self.car.Fy_f(Ux_final,Uy_final,r_final,delta_final,Fx)
                Fy_r_final = self.car.Fy_r(Ux_final,Uy_final,r_final,delta_final,Fx)
                cost += (cost_weights['Fx']/ds) * (Fx_bar_initial - Fx)**2

            # --------- Stuff that breaks things (Friction Limits for real world experiments) -----
            # opti.subject_to(self.Fy_f[n] == self.car.Fy_f(Ux,Uy,r,delta,Fx))
            # opti.subject_to(self.Fy_r[n] == self.car.Fy_r(Ux,Uy,r,delta,Fx))
            # opti.subject_to(self.Fx_f[n] == self.car.Fx_f(Fx))
            # opti.subject_to(self.Fx_r[n] == self.car.Fx_r(Fx))
            # opti.subject_to(self.Fx_f[n]**2 + self.Fy_f[n]**2 <= (input_constraints['mu_lim']*self.car.Fz_f(Ux,Fx))**2 + (self.Fe_f[n])**2)
            # opti.subject_to(self.Fx_r[n]**2 + self.Fy_f[n]**2 <= (input_constraints['mu_lim']*self.car.Fz_r(Ux,Fx))**2 + (self.Fe_r[n])**2) 
            # ---------------------------------------------------------------------
        
        # ===================== Point Mass Model =====================================
        for m in range(self.M):
            state_pm = self.state_pm[:,m]
            state_pm_next = self.state_pm[:,m+1]
            input_pm = self.action_pm[:,m]
            V_bar = state_pm[self.point_mass.state.index('V')]
            ey_bar = state_pm[self.point_mass.state.index('ey')]
            epsi_bar = state_pm[self.point_mass.state.index('epsi')]
            Fx_bar = input_pm[self.point_mass.input.index('Fx')]
            Fy_bar = input_pm[self.point_mass.input.index('Fy')]
            
            # -------------------- Constraints --------------------------------------------------
            # state limits
            opti.subject_to(V_bar >= state_pm_constraints['V_bar_min'])
            
            # input limits
            opti.subject_to(Fx <= Peng / V_bar)
            
            # Discretization (Going on for dt with displacement snapshot) 
            curvature = self.car.track.get_curvature(state_pm_next[self.car.state.index('s')])
            opti.subject_to(self.ds_pm[m] == ds_bar)
            
            # Model dynamics 
            opti.subject_to(state_pm_next == self.point_mass.spatial_transition(state_pm,input_pm,curvature,ds_bar)) 
            
            # friction limits TODO

            # -------------------- Stage Cost -------------------------------------
            cost += ca.if_else(ey_bar < state_pm_constraints['ey_bar_min'], # 3) road boundary intrusion
                       cost_weights['boundary']*ds_bar*(ey_bar - state_pm_constraints['ey_bar_min'])**2, 0)
            
            cost += ca.if_else(ey_bar > state_pm_constraints['ey_bar_max'], # 3) road boundary intrusion
                       cost_weights['boundary']*ds_bar*(ey_bar - state_pm_constraints['ey_bar_max'])**2, 0)
            
            cost += cost_weights['deviation']*ds_bar*(ey_bar**2) # 4) deviation from road descriptor path
            
            # cost += cost_weights['friction']*((self.Fe_f[self.N-1+m]**2)**2 + (self.Fe_r[self.N-1+m]**2)**2) # friction limit
            
            if m < self.M-1: # 5) Slew Rate
                next_input_pm = self.action_pm[:,m+1]
                Fy_bar_next = next_input_pm[self.point_mass.input.index('Fy')]
                Fx_bar_next = next_input_pm[self.point_mass.input.index('Fx')]
                cost += cost_weights['Fy']*(1/ds_bar)*(Fy_bar_next - Fy_bar)**2
                cost += cost_weights['Fx']*(1/ds_bar)*(Fx_bar_next - Fx_bar)**2 

        # -------------------- Terminal Cost -----------------------------------------
        if self.M > 0:
            final_state = self.state_pm[:,-1]
            final_model = self.point_mass
            final_speed = final_state[final_model.state.index('V')]
        else:
            final_state = self.state[:,-1]
            final_model = self.car
            final_speed = final_state[final_model.state.index('Ux')]
        cost += ca.if_else(final_speed >= state_constraints['max_speed'],cost_weights['speed']*(final_speed - state_constraints['max_speed'])**2, 0) # excessive speed
        cost += cost_weights['time']*(final_state[final_model.state.index('t'),-1])**2 # final cost (minimize time)
        cost += cost_weights['ey']*final_state[final_model.state.index('ey'),-1]**2 # final cost (minimize terminal lateral error) hardcodato
        cost += cost_weights['epsi']*final_state[final_model.state.index('epsi'),-1]**2 # final cost (minimize terminal course error) hardcodato
        opti.minimize(cost)
        return opti
import time
from controllers.controller import Controller
import numpy as np
from modeling.kinematic_car import KinematicCar
import logging
from modeling.util import wrap
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from modeling.track import Track

class RacingSimulation():   
    def __init__(self, car: KinematicCar, controller: Controller):
        self.car = car
        self.controller = controller
        logging.basicConfig(
            filename="test.log", 
            filemode='w', 
            level=logging.INFO, 
            format='%(message)s'
        )
        
    def run(self, animate: bool = True):
        
        # Logging containers
        state_traj = [self.car.state] # state trajectory (logging)
        action_traj = [] # input trajectory (logging)
        elapsed = [] # elapsed times
        state_preds = [] # state predictions for each horizon
        
        # Initializing simulation
        state = state_traj[0] 
        s = 0
        kappa = np.array([self.car.track.waypoints[i].kappa for i in range(self.controller.horizon)])
        
        # Starting Simulation
        while True:
            if s > self.car.track.length: break
            # computing control signal
            start = time.time()
            action, state_prediction = self.controller.command(state, kappa)
            elapsed.append(time.time() - start)
            
            # computing path geometry for next horizon
            kappa = np.array([self.car.track.waypoints[i].kappa for i in range(self.controller.horizon)])
            
            # applying control signal
            state = self.car.drive(action)
            s = state.s
            state.psi = wrap(state.psi)
            
            # logging
            state_traj.append(state)
            action_traj.append(action)
            state_preds.append(state_prediction)
            logging.info(self.car.state)
            logging.info(self.car.current_waypoint)
        
        logging.shutdown()
        if animate:
            self.animate(state_traj, action_traj, state_preds, elapsed)   
        
    
    def animate(self, state_traj: list, input_traj: list, state_preds: list, elapsed: list):
        assert isinstance(state_traj,list), "State traplotjectory has to be a list"
        assert isinstance(state_traj,list), "Input trajectory has to be a list"

        # simulation params
        N = len(input_traj)
        x_y = np.array(state_traj)[:,:2]
        v_psi_delta = np.array(state_traj)[:,2:5] # taking just v,psi,delta TO
        a_w = np.array(input_traj)
        
        # figure params
        grid = GridSpec(2, 2, width_ratios=[2, 1])
        ax_large = plt.subplot(grid[:, 0])
        ax_small1 = plt.subplot(grid[0, 1])
        ax_small2 = plt.subplot(grid[1, 1])
        state_max = max(v_psi_delta.min(), v_psi_delta.max(), key=abs) # for axis limits
        input_max = max(a_w.min(), a_w.max(), key=abs) # for axis limits
        
        # fig titles
        lap_time = plt.gcf().text(0.4, 0.95, 'Laptime', fontsize=16, ha='center', va='center')
        elapsed_time = plt.gcf().text(0.4, 0.85, 'Mean computation time', fontsize=16, ha='center', va='center')
        
        def update(i):
            state = state_traj[i]
            
            lap_time.set_text(f"Lap time: {state.t:.2f}") 
            
            if i%5 == 0:
                elapsed_time.set_text(f"Mean computation time: {np.mean(elapsed[i-5:i]):.2f}")
            
            ax_large.cla()
            ax_large.set_aspect('equal')
            self.car.plot(ax_large, state)
            ax_large.plot(x_y[:i, 0],x_y[:i, 1],"k")
            
            # Plot track
            if self.car.track is not None:
                self.car.track.plot(ax_large, display_drivable_area=False)
                
            # Plot state predictions of MPC
            if state_preds is not None:
                preds = state_preds[i]
                ax_large.plot(preds[0,:], preds[1,:],"go")  
            
            ax_small1.cla()
            ax_small1.axis((0, N, -state_max*1.1, state_max*1.1))
            ax_small1.plot(v_psi_delta[:i, :], '-', alpha=0.7,label=['v',r'$\psi$',r'$\delta$'])
            ax_small1.legend()

            ax_small2.cla()
            ax_small2.axis((0, N, -input_max*1.1, input_max*1.1))
            ax_small2.plot(a_w[:i, :], '-', alpha=0.7,label=['a','w'])
            ax_small2.legend()

        animation = FuncAnimation(
            fig=plt.gcf(), func=update, 
            frames=N, interval=0.01, 
            repeat=False, repeat_delay=5000
        )
        
        # Get the current figure manager
        fig_manager = plt.get_current_fig_manager()
        # Maximize the window
        fig_manager.window.showMaximized()
        plt.show()  
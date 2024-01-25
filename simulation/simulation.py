import time
from controllers.controller import Controller
import numpy as np
from modeling.racing_car import KinematicCar
import logging

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
        
    def run(self):
        
        # Initiating Simulation
        state_traj = [self.car.state] # state trajectory (logging)
        action_traj = [] # input trajectory (logging)
        state = state_traj[0]
        elapsed = []
        state_preds = []
        
        s = 0
        
        # Starting Simulation
        while True:
            if s > self.car.track.length - 0.1: break
            # computing control signal
            start = time.time()
            action, state_prediction = self.controller.command(state)
            
            state_preds.append(state_prediction)
            
            # print(f"Input: {action}")
            
            elapsed.append(time.time() - start)
            
            # applying control signal
            state = self.car.drive(action)
            
            s = state.s
            
            # print(f"State: {state}")
            
            logging.info(self.car.state)
            logging.info(self.car.current_waypoint)
            
            # logging
            state_traj.append(state)
            action_traj.append(action)
        
        print(f"Mean time per horizon: {np.mean(elapsed)}")
        logging.shutdown()
        return np.array(state_traj), np.array(action_traj), np.array(state_preds)
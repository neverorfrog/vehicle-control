import time
from controllers.controller import Controller
import numpy as np
from modeling.racing_car import RacingCar

class RacingSimulation():   
    def __init__(self, car: RacingCar, controller: Controller):
        self.car = car
        self.controller = controller
        
    def run(self, N: int):
        
        # Initiating Simulation
        s_traj = [self.car.state] # state trajectory (logging)
        i_traj = [] # input trajectory (logging)
        s_k = s_traj[0]
        elapsed = []
        
        # Starting Simulation
        for _ in range(N):
            # computing control signal
            start = time.time()
            i_k = self.controller.command(s_k)
            elapsed.append(time.time() - start)
            
            # applying control signal
            s_k = self.car.drive(i_k)
            
            # logging
            s_traj.append(s_k)
            i_traj.append(i_k)
        
        print(f"Mean time per horizon: {np.mean(elapsed)}")
        return np.array(s_traj), np.array(i_traj)
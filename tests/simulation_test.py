import sys
sys.path.append("..")

from model import *
import numpy as np

# Simulation of the system by recursively calling the discrete time dynamics
# functions with constant velocity (both translational and rotational). This
# means we will make a circle at a certain speed.

# The output will be a state/input trajectory

PARAMS={
    'dt': 0.03,
    'integration_steps': 10
}

def simulate():
    model = DifferentialDrive(PARAMS)
    N = int(ca.floor((2*ca.pi)/PARAMS['dt'])) # number of steps (enough to do a full turn)
    state_traj  = np.zeros((model.state_len, N+1)) # preallocate array for the state trajectory
    input_traj  = np.ones((model.input_len, N)) # define (constant) action trajectory
    NUMBER_OF_TURNS = 1
    input_traj[0,:] *= 1 # translational velocity
    input_traj[1,:] *= NUMBER_OF_TURNS # rotational velocity (factor is number of turns)

    # initial state
    state_traj[:, 0]  = np.array([0., -0.5, 0.])

    # forward simulation
    for k in range(N):
        state_k = state_traj[:, k]
        input_k = input_traj[:, k]
        state_traj[:, [k+1]] = model.step(state_k,input_k)
        
    print(state_traj[:,-1])
    print(ca.pi * NUMBER_OF_TURNS)
    
    return state_traj, input_traj, N

# Now this trajectory will be animated

from utils import animate

state_traj, input_traj, N = simulate()
animation = animate(state_traj, input_traj, state_labels=['x','y','theta'], input_labels=['v','w'])

    
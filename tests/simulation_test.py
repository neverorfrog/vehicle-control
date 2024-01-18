import sys
sys.path.append("..")

from modeling.robot import *
from controllers.controller import Controller
import numpy as np
from simulation.simulation import Simulation

# Simulation of the system by recursively calling the discrete time dynamics
# functions with constant velocity (both translational and rotational). This
# means we will make a circle at a certain speed.

# The output will be a state/input trajectory

dt = 0.1

def simulate():
    robot = DifferentialDrive()
    N = int(ca.floor((2*ca.pi)/dt)) # number of steps (enough to do a full turn)
    q_traj  = np.zeros((N+1, robot.q_len)) # preallocate array for the state trajectory
    u_traj  = np.ones((N, robot.u_len)) # define (constant) action trajectory
    q_traj[0, :]  = np.array([0., -0.5, 0.]) # initial state
    
    # # Arbitrary velocity
    NUMBER_OF_TURNS = 2
    u_traj[:,0] *= 1 # translational velocity
    u_traj[:,1] *= NUMBER_OF_TURNS # rotational velocity (factor is number of turns)

    # forward simulation
    controller = Controller() #stub
    loop = Simulation(robot, controller, dt)
    for k in range(N):
        q_k = q_traj[k, :]
        u_k = u_traj[k, :]
        next_q, next_qd = loop.step(q_k,u_k)
        q_traj[[k+1],:] = next_q
        
    # assert to test simulation result
    expected_final_state = [q_traj[0,0], q_traj[0,1], 2 * ca.pi * NUMBER_OF_TURNS]
    assert((np.abs(q_traj[-1,:] - expected_final_state) < 0.5).all())

    return q_traj, u_traj, N

# Now this trajectory will be animated
from simulation.plotting import animate
q_traj, u_traj, N = simulate()
animation = animate(q_traj, u_traj, state_labels=['x','y','theta'], input_labels=['v','w'])

    
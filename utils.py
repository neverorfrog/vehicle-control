from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import casadi as ca

def RK4(state, input, state_dot, dt, N_steps=1):
    '''RK4 integrator.
    
    state, input:   casadi expression that have been used to define the dynamics state_dot
    state_dot:      casadi expr defining the rhs of the ode
    dt:             integration interval
    N_steps:        number of integration steps per integration interval, default:1
    '''

    h = dt/N_steps
    current_state = state
    model = ca.Function('xdot', [state, input], [state_dot])

    for _ in range(N_steps):
        k_1 = model(current_state, input)
        k_2 = model(current_state + (dt/2)*k_1,input)
        k_3 = model(current_state + (dt/2)*k_2,input)
        k_4 = model(current_state + dt*k_3,input)

        current_state += (1/6) * (k_1 + 2 * k_2 + 2 * k_3 + k_4) * h

    return current_state

def plot_trajectories(state_trajs, input_trajs, labels, state_labels):

    if not isinstance(state_trajs, list):
        state_trajs = [state_trajs]
        input_trajs = [input_trajs]
        labels = [labels]
    
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    N = input_trajs[0].shape[1]
    ts = np.arange(0, N+1)
    
    # latexify()

    plt.figure(figsize=(5, 8))
    
    state_len = state_trajs[0].shape[0]
    for j in range(state_len):
        plt.subplot(state_len+1, 1, j+1)
        for i, x_traj in enumerate(state_trajs):
            print(x_traj)
            plt.plot(ts, x_traj[0, :].T, '-', alpha=0.7, color=colors[i], label=labels[i])
        plt.ylabel(rf'${state_labels[j]}_k$')
        plt.grid()
        plt.legend()

    plt.subplot(state_len+1, 1, state_len+1)
    for i, u_traj in enumerate(input_trajs):
        plt.step(ts[:-1], u_traj.T, alpha=0.7, color=colors[i], label=labels[i], where='post')
    plt.grid()
    plt.xlabel(r'time step $k$')
    plt.ylabel(r'$u_k$')
    plt.legend()
    
if __name__=="__main__":
    N = 5
    state_traj  = np.zeros((2, N+1))
    input_traj  = 0.0*np.ones((1, N))
    plot_trajectories(state_traj, input_traj, 'simulated',['x','y'])
    plt.show()
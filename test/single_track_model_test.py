import sys
sys.path.append("..")

from simulation.simulator import *
from environment import *
from environment.track import *
from environment.trajectory import *
from model.single_track_model import *

wp = np.array([[-2,0],[2,0],[2,2],[-2,2],[-2,0],[-0.5,0]])
reference = Track(wp_x=wp[:,0], wp_y=wp[:,1], resolution=0.03,smoothing=25,width=0.4)
robot = SingleTrack()
circle = Circle(freq = 0.02)
# for offline plotting

# control loop
def step( q_k, u_k):
    """
        - Given current (kth) q and u
        - Applies it for dt (given in Robot construction)
        - Return numpy array of next q
    """
    #print(q_k, u_k)
    next_q = discrete_ode(q_k,u_k).full().squeeze()
    #print(next_q)
    next_qd = robot.transition_function(next_q, u_k).full().squeeze()
    #print(next_q)
    #input("-----")
    return next_q, next_qd
T=10
q0 = np.array([10,0,0,0,0,0,0])
qd0=np.array([0,0,0,0])
dt = 0.25
q_traj = [q0]
qd_traj = [qd0]
u_traj = []
time = [0]

# control loop initialization
q_k = q_traj[-1]
qd_k = qd_traj[-1]

discrete_ode = ca.Function('discrete_ode', [robot.q,robot.u], [robot.RK4(dt)])


while True:
    time.append(time[-1] + dt)
    if time[-1] >= T: break

    if time[-1]<3:cmd = 1
    elif time[-1]<5.75 :cmd=-0
    else: cmd =0
    u_k = np.array([1,cmd])
    
    q_k, qd_k = step(q_k,u_k)
    
    # logging
    q_traj.append(q_k)
    qd_traj.append(qd_k)
    u_traj.append(u_k)
    #ref_traj.append(ref_k['p'])
q_traj = np.array(q_traj)
u_traj = np.array(u_traj)
print(q_traj.shape)
print("Ux_TRAJ:", q_traj[:,0])
print("___________________________________")
print("Uy_TRAJ:", q_traj[:,1])
print("___________________________________")
print("R_PSI_TRAJ:", q_traj[:,2])
print("___________________________________")
print("S:", q_traj[:,3])
print("___________________________________")
print("E:", q_traj[:,4])
print("___________________________________")
print("D_PSI_TRAJ:", q_traj[:,5])
print("___________________________________")
print("Delta_traj:", q_traj[:,6])
print("___________________________________")
print("Commanded force _traj:", u_traj[:,0])
print("___________________________________")
print("Commanded steering _traj:", u_traj[:,1])
print("___________________________________")
    
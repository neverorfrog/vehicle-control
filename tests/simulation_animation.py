import sys
sys.path.append("..")

from utils import animate
from model_simulation import simulate
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

state_traj, input_traj, N = simulate()
update = animate(state_traj, input_traj, state_labels=['x','y','theta'], input_labels=['v','w'])
animation = FuncAnimation(fig=plt.gcf(), func=update, frames=N+1, repeat=True, interval=0.01)
plt.show()
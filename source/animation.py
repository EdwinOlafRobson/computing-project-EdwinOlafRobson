import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D 


def animate_framerate(trajectory_positions, interval = 30):

    " Animates to a given framerate"

    steps, N, _ = trajectory_positions.shape

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scat = ax.scatter([], [], [])
    all_pos = trajectory_positions.reshape(-1, 3)
    mins = all_pos.min(axis=0)
    maxs = all_pos.max(axis=0)

    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])

    def update(frame):
        pos = trajectory_positions[frame]

        scat._offsets3d = (pos[:,0], pos[:,1], pos[:,2])
        return scat,

    anim = FuncAnimation(fig, update, frames=steps, interval=interval)

    plt.show()
    return anim

def animate_length(trajectory_positions, length=10):
    
    """ Animates lasts `length` seconds at 30 fps. """

    fps = 30
    interval = 1000 / fps  

    steps, N, _ = trajectory_positions.shape
    total_frames = int(fps * length)

    frame_indices = np.linspace(0, steps - 1, total_frames).astype(int)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scat = ax.scatter([], [], [])

    all_pos = trajectory_positions.reshape(-1, 3)
    mins = all_pos.min(axis=0)
    maxs = all_pos.max(axis=0)

    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])

    def update(frame):
        pos = trajectory_positions[frame_indices[frame]]
        scat._offsets3d = (pos[:, 0], pos[:, 1], pos[:, 2])
        return scat,

    anim = FuncAnimation(fig, update, frames=total_frames, interval=interval)

    plt.show()
    return anim
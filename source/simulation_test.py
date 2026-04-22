import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D 


import acceleration_calculation.accelerations as accelerations
import animation
import system
import time_step
import physical_tests

trajectory_positions, trajectory_velocities = system.simulate(
    system.gravitational_constant_random_position_no_net_velocity(N=40, box_size=6, max_velocity=2),
    step_function= time_step.leapfrog_step,
    acceleration_function = accelerations.gravitational_fmm_acceleration,
    steps = 480,
    dt = 0.01,
    G = 1,
    softening = 1e-5,

)

animation.animate_framerate(trajectory_positions)


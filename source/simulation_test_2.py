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
    system.gravitational_uniform_position_stationary_velocity(N=27, box_size=3),
    step_function= time_step.leapfrog_step,
    acceleration_function = accelerations.gravitational_fmm_simple_acceleration,
    steps = 100,
    dt = 0.001,
    G = 1,
    softening = 1e-5,

)

animation.animate_framerate(trajectory_positions)
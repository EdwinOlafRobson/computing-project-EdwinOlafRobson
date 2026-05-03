import numpy as np
import matplotlib.pyplot as ply

import animation 
import system
import time_step
from acceleration_calculation import accelerations


trajectory_position, trajectory_velocity = system.simulate(
    system.gravitational_constant_random_position_no_net_velocity(30,5,2),
    time_step.leapfrog_step,
    accelerations.gravitational_fmm_acceleration,
    500,
    G = 5, 

 
    
)
animation.animate_length(trajectory_position, 10)

np.save("source/Results/animation/fmm.npy", trajectory_position)
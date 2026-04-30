import numpy as np

import acceleration_calculation.accelerations as accelerations
import system
import time_step
import physical_tests
import animation

trajectory_positions, trajectory_velocities = system.simulate(
    system.gravitational_random_position_random_velocity(50,7,0),
    time_step.simple_step,
    accelerations.gravitational_fmm_acceleration,
    60,
    0.01,
    5)

animation.animate_length(trajectory_positions, 5)
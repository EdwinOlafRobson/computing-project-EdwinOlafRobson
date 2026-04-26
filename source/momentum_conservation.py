import numpy as np
import copy
import os

import acceleration_calculation.accelerations as accelerations
import system
import time_step
import physical_tests

virials = ["0.001", "1", "1000"]

steps = 1000
dt = 0.01
softening = 1e-5

for virial in virials:

    virial_vmax = np.load(f"source/Results/Virial_vmax/box_5_target_{virial}.npy")

    number_of_particles = int(virial_vmax[0, 0])
    maximum_velocity = virial_vmax[0, 1]

    initial_system_state = system.gravitational_constant_random_position_no_net_velocity(
        N=number_of_particles,
        box_size=5,
        max_velocity=maximum_velocity
    )

    total_normalised_momenta = []

    for theta in [0, 0.5, 1]:
        trajectory_positions, trajectory_velocities = system.simulate(
            copy.deepcopy(initial_system_state),
            step_function=time_step.rk4_step,
            acceleration_function=accelerations.gravitational_barnes_hut_acceleration,
            steps=steps,
            dt=dt,
            softening=softening,
            threshold=theta
        )

        total_normalised_momentum = physical_tests.total_normalised_momentum(trajectory_velocities)
        total_normalised_momenta.append(total_normalised_momentum)

    trajectory_positions, trajectory_velocities = system.simulate(
        copy.deepcopy(initial_system_state),
        step_function=time_step.rk4_step,
        acceleration_function=accelerations.gravitational_vectorised_acceleration,
        steps=steps,
        dt=dt,
        softening=softening
    )

    total_normalised_momentum = physical_tests.total_normalised_momentum(trajectory_velocities)
    total_normalised_momenta.append(total_normalised_momentum)

    trajectory_positions, trajectory_velocities = system.simulate(
        copy.deepcopy(initial_system_state),
        step_function=time_step.rk4_step,
        acceleration_function=accelerations.gravitational_fmm_acceleration,
        steps=steps,
        dt=dt,
        softening=softening
    )

    total_normalised_momentum = physical_tests.total_normalised_momentum(trajectory_velocities)
    total_normalised_momenta.append(total_normalised_momentum)

    total_normalised_momenta = np.array(total_normalised_momenta)

    save_folder = f"source/Results/total momentum/{virial}"
    os.makedirs(save_folder, exist_ok=True)

    np.save(f"{save_folder}/momentum.npy", total_normalised_momenta)

    print(f"Virial = {virial} test complete")
import numpy as np

import acceleration_calculation.accelerations as accelerations
import system
import time_step
import physical_tests


virial_target = 1000
tolerance = 1e-3  

N_values = [10, 20, 50, 80, 100, 200, 300, 500, 700, 1000]

vmax_results = []


def compute_virial(N, vmax):
    trajectory_positions, trajectory_velocities = system.simulate(
        system.gravitational_constant_random_position_no_net_velocity(
            N=N, box_size=5, max_velocity=vmax
        ),
        step_function=time_step.leapfrog_step,
        acceleration_function=accelerations.gravitational_vectorised_acceleration,
        steps=1,
        dt=0.0001,
        softening=0
    )

    return physical_tests.virial_ratio(trajectory_positions, trajectory_velocities)


def find_vmax_for_target(N, target):
    v_low = 0.0
    v_high = 1.0  


    while compute_virial(N, v_high) < target:
        v_high *= 2

    for _ in range(30): 
        v_mid = 0.5 * (v_low + v_high)
        r_mid = compute_virial(N, v_mid)

        if abs(r_mid - target) / target < tolerance:
            return v_mid

        if r_mid < target:
            v_low = v_mid
        else:
            v_high = v_mid

    return v_mid  


# Main loop
for N in N_values:
    print(f"Processing N = {N}...")
    vmax = find_vmax_for_target(N, virial_target)
    vmax_results.append(vmax)

print("\nVmax values:")
print(vmax_results)

np.save(f"source/Results/Virial_vmax/box_5_target_{virial_target}", np.column_stack((N_values, vmax_results)))
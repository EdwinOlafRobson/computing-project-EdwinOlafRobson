import numpy as np
import os
import time

import acceleration_calculation.accelerations as accelerations
import system
import time_step

target_001 = np.load("source/Results/Virial_vmax/box_5_target_0.001.npy")
target_001_v_max = target_001[:, 1]
target_001_N = target_001[:, 0]

target_1 = np.load("source/Results/Virial_vmax/box_5_target_1.npy")
target_1_v_max = target_1[:, 1]
target_1_N = target_1[:, 0]

target_1000 = np.load("source/Results/Virial_vmax/box_5_target_1000.npy")
target_1000_v_max = target_1000[:, 1]
target_1000_N = target_1000[:, 0]

MAX_TIME = 2.0
MIN_REPEATS = 5


def benchmark_frame_time(N, v_max, virial_label, method_label,
                         step_function, acceleration_function,
                         dt=0.0001, softening=0, box_size=5, **kwargs):

    times = []
    start_total = time.perf_counter()

    i = 0
    while True:
        t0 = time.perf_counter()

        system.simulate(
            system.gravitational_constant_random_position_no_net_velocity(
                N=N,
                box_size=box_size,
                max_velocity=v_max
            ),
            step_function=step_function,
            acceleration_function=acceleration_function,
            steps=1,
            dt=dt,
            softening=softening,
            **kwargs
        )

        t1 = time.perf_counter()
        times.append(t1 - t0)
        i += 1

        if i >= MIN_REPEATS and (time.perf_counter() - start_total) >= MAX_TIME:
            break

    result = np.array([np.mean(times), np.std(times)])

    folder = f"source/Results/{method_label}/{virial_label}"
    os.makedirs(folder, exist_ok=True)

    np.save(f"{folder}/N_{N}_vmax_{v_max}.npy", result)

    print(f"{method_label} | virial={virial_label} | N={N} | v_max={v_max}")

    return result


def run_all(method_label, step_function, acceleration_function, **kwargs):

    datasets = [
        ("0.001", target_001_N, target_001_v_max),
        ("1", target_1_N, target_1_v_max),
        ("1000", target_1000_N, target_1000_v_max),
    ]

    for virial_label, Ns, vels in datasets:
        for N, v_max in zip(Ns, vels):
            benchmark_frame_time(
                N=int(N),
                v_max=float(v_max),
                virial_label=virial_label,
                method_label=method_label,
                step_function=step_function,
                acceleration_function=acceleration_function,
                **kwargs
            )


print("Starting benchmark suite...")


run_all(
    method_label="vectorised",
    step_function=time_step.leapfrog_step,
    acceleration_function=accelerations.gravitational_vectorised_acceleration
)

run_all(
    method_label="fmm",
    step_function=time_step.leapfrog_step,
    acceleration_function=accelerations.gravitational_fmm_acceleration
)

run_all(
    method_label="bh_0",
    step_function=time_step.leapfrog_step,
    acceleration_function=accelerations.gravitational_barnes_hut_acceleration,
    threshold=0
)

run_all(
    method_label="bh_0.5",
    step_function=time_step.leapfrog_step,
    acceleration_function=accelerations.gravitational_barnes_hut_acceleration,
    threshold=0.5
)

run_all(
    method_label="bh_1",
    step_function=time_step.leapfrog_step,
    acceleration_function=accelerations.gravitational_barnes_hut_acceleration,
    threshold=1
)

print("Benchmark complete")
import numpy as np

def leapfrog_step(system, acceleration_function, dt):
    
    ''' Velocity Verlet integrator. 1/2 steps velocity around position. '''

    acc = acceleration_function(system)

    system.velocities += 0.5 * dt * acc

    system.positions += dt * system.velocities

    acc_new = acceleration_function(system)

    system.velocities += 0.5 * dt * acc_new

    return system


def simple_step(system, acceleration_function, dt):

    ''' Naieve implementation. '''

    acc = acceleration_function(system)

    system.velocities += dt * acc

    system.positions += dt * system.velocities

    return system


def copy_system(system, positions, velocities):

    ''' Creates a true copy of system, used for Runge Kutta. '''

    return type(system)(
        positions.copy(),
        velocities.copy(),
        system.masses.copy(),
        system.charges.copy()
    )



def rk4_step(system, acceleration_function, dt):

    ''' Runge Kutta with 4 steps. '''

    x0 = system.positions
    v0 = system.velocities


    a1 = acceleration_function(system)
    k1_v = v0
    k1_a = a1


    sys2 = copy_system(system,
                       x0 + 0.5 * dt * k1_v,
                       v0 + 0.5 * dt * k1_a)
    a2 = acceleration_function(sys2)
    k2_v = sys2.velocities
    k2_a = a2


    sys3 = copy_system(system,
                       x0 + 0.5 * dt * k2_v,
                       v0 + 0.5 * dt * k2_a)
    a3 = acceleration_function(sys3)
    k3_v = sys3.velocities
    k3_a = a3


    sys4 = copy_system(system,
                       x0 + dt * k3_v,
                       v0 + dt * k3_a)
    a4 = acceleration_function(sys4)
    k4_v = sys4.velocities
    k4_a = a4


    system.positions = x0 + dt * (k1_v + 2*k2_v + 2*k3_v + k4_v) / 6
    system.velocities = v0 + dt * (k1_a + 2*k2_a + 2*k3_a + k4_a) / 6

    return system
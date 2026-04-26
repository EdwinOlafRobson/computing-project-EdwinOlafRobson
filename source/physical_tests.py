import numpy as np
import matplotlib.pyplot as plt

def kinetic_energy(velocities):
    kinetic_energy = 0.5 * np.sum(velocities ** 2)
    return kinetic_energy


def gravitational_potential_energy(positions, G=1, softening=1e-5):

    '''Vectorised GPE using same structure as acceleration.'''

   
    dx = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist_sq = np.sum(dx**2, axis=2) + softening**2
    inverse_distance = 1.0 / np.sqrt(dist_sq)

    np.fill_diagonal(inverse_distance, 0)

    potential_energy = -G * np.sum(np.triu(inverse_distance, k=1))

    return potential_energy


def gravitational_potential_energy_time(trajectory_positions, G=1, softening=1e-5):
 
    '''Vectorised GPE using same structure as acceleration. 
    Vecorised over time, but uses LOTS of memeory'''

    dx = trajectory_positions[:, :, np.newaxis, :] - trajectory_positions[:, np.newaxis, :, :]

    dist_sq = np.sum(dx**2, axis=3) + softening**2

    inverse_distance = 1.0 / np.sqrt(dist_sq)

    # zero self-interaction for every timestep
    T, N, _ = inverse_distance.shape
    idx = np.arange(N)
    inverse_distance[:, idx, idx] = 0

    # sum over unique pairs
    U = -G * np.sum(np.triu(inverse_distance, k=1), axis=(1,2))

    return U   


def total_energy(trajectory_positions, trajectory_velocities):

    time = trajectory_positions.shape[0]
    PE = np.zeros(time)
    for i in range(time):
        PE[i] = gravitational_potential_energy(trajectory_positions[i])

    KE = 0.5*np.sum(trajectory_velocities**2, axis=(1,2))
    
    return PE + KE

def virial(trajectory_positions, trajectory_velocities):
    
    time = trajectory_positions.shape[0]
    PE = np.zeros(time)
    for i in range(time):
        PE[i] = gravitational_potential_energy(trajectory_positions[i])

    KE = 0.5*np.sum(trajectory_velocities**2, axis=(1,2))
    
    return PE + 2*KE


def virial_ratio(trajectory_positions, trajectory_velocities):
    
    time = trajectory_positions.shape[0]
    PE = np.zeros(time)
    for i in range(time):
        PE[i] = gravitational_potential_energy(trajectory_positions[i])

    KE = 0.5*np.sum(trajectory_velocities**2, axis=(1,2))

    return 2*KE / abs(PE)

def total_normalised_momentum(trajectory_velocities):

    ''' Returns the sum of the squares of the momentum'''

    momentum = np.sum(trajectory_velocities, axis = 1)
    momentum_norm = np.linalg.norm(momentum, axis = 1)
    return momentum_norm


def plot(virial):
    time = np.arange(len(virial))

    plt.plot(time, virial, c="r")
    plt.xlabel("Time")
    plt.ylabel("Energy")
    plt.axhline(y=0, linestyle='--')
    plt.show()
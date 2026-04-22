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


def virial_n_filtering(virial_array, n):

    """ Removes the n biggest unphysical discontinuities in the virial."""

    virial_copy = virial_array.copy()
    
    diffs = np.diff(virial_copy)
    pos_idx = np.where(diffs > 0)[0]
    if len(pos_idx) == 0:
        return virial_copy 
    
    largest = pos_idx[np.argsort(diffs[pos_idx])[-n:]]
    
    largest = np.sort(largest)
    

    for i in largest:
        jump = virial_copy[i+1] - virial_copy[i]
        virial_copy[i+1:] -= jump
    
    return virial_copy 
    
def virial_filter_adaptive(virial, k=3.0, smooth=True):
    v = virial.copy()
    N = len(v)
    diffs = np.diff(v)
    

    mu = np.mean(diffs)
    sigma = np.std(diffs)
    threshold = mu + k * sigma
  
    jump_indices = np.where(diffs > threshold)[0]
    
    if len(jump_indices) == 0:
        return v - v[-1]
    
    for i in jump_indices:
        jump = v[i+1] - v[i]
        tail_len = N - (i + 1)
        
        if tail_len <= 0:
            continue
        
        if smooth:

            ramp = np.linspace(0, 1, tail_len)
            v[i+1:] -= jump * ramp
        else:

            v[i+1:] -= jump
    v -= v[-1]
    
    return v



def plot(virial):
    time = np.arange(len(virial))

    plt.plot(time, virial, c="r")
    plt.xlabel("Time")
    plt.ylabel("Energy")
    plt.axhline(y=0, linestyle='--')
    plt.show()
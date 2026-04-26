import numpy as np
from . import barnes_hut 
from . import fast_multipole


def gravitational_pairwise_acceleration(system, G=1, softening=1e-5):
    
    ''' The Naieve approach - iterating over every pair. '''

    N = system.N
    positions = system.positions

    acceleration = np.zeros_like(positions)

    for i in range(N):
        for j in range (i+1, N):
            vector_ij = positions[i] - positions[j]
            length_ij_squared = np.dot(vector_ij, vector_ij) + softening**2
            length_ij = np.sqrt(length_ij_squared)
            length_ij_cubed = length_ij_squared * length_ij

            # Not pre-normalising the vector saves 1 division.

            force = (-G / (length_ij_cubed)) * vector_ij

            # Uses Newton's 3rd law to ~1/2 the computation.

            acceleration[i] += force
            acceleration[j] += -force

    return acceleration


def gravitational_vectorised_acceleration(system, G=1, softening=1e-5):

    '''Naive approach with Numpy broadcasting.'''
    # ~320 times faster than Numpy loops.

    positions = system.positions

    # Numpy broadcasting instead of looping over index.
    dx = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]   

    dist_sq = np.sum(dx**2, axis=2) + softening**2    
    inverse_distance_cubed = 1.0 / (dist_sq * np.sqrt(dist_sq))

    np.fill_diagonal(inverse_distance_cubed, 0)

    forces_div_masses = -G * dx * inverse_distance_cubed[:, :, np.newaxis] 

    acceleration = np.sum(forces_div_masses, axis=1)

    return acceleration




def gravitational_barnes_hut_acceleration(system, G=1, softening=1e-5, threshold=0.5):

    """ Barnes-Hut method. """

    positions = system.positions
    N = system.N

    acc = np.zeros_like(positions)
    
    # Root node.
    min_pos = np.min(positions, axis=0)
    max_pos = np.max(positions, axis=0)
    centre = 0.5 * (min_pos + max_pos)
    size = np.max(max_pos - min_pos)
    root = barnes_hut.OctTreeNode(centre, size / 2)

    # Put the particles in the Octree.
    for i in range(N):
        barnes_hut.insert(root, system, i)

    # Masses and COM of all nodes.
    barnes_hut.compute_mass_distribution(root, system)
    
    for i in range(N):
        acc[i] = barnes_hut.compute_gravitational_acceleration(root, system, i, threshold, G, softening)

    return acc



def gravitational_fmm_acceleration(system, G=1, softening=1e-5):

    """ Second order Fast Multipole Methid implementation."""

    # Bounding box
    positions = system.positions
    minimum_position = np.min(positions, axis=0)
    maximum_position = np.max(positions, axis=0)
    centre = 0.5 * (minimum_position + maximum_position)
    size = np.max(maximum_position - minimum_position)
 
    root = fast_multipole.OctTreeNode(centre, size / 2.0)
    root.depth = 0
 
    for particle_index in range(system.N):
        fast_multipole.insert(root, system, particle_index)
 

    fast_multipole.upward_pass(root, system)
    fast_multipole.build_interaction_lists(root)
    fast_multipole.downward_pass(root)
 
    return fast_multipole.evaluate_leaves(root, system, G, softening)
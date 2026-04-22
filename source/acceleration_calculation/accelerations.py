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

    # Barnes-Hut method. 

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


def gravitational_fmm_acceleration(system, G=1, softening=1e-5, expansion_order=4):
    """Compute gravitational accelerations for all particles using the
    Fast Multipole Method (FMM).

    Parameters
    ----------
    system          : object with attributes
                          positions  – (N, 3) float array
                          N          – number of particles
    G               : gravitational constant (default 1)
    softening       : Plummer softening length to avoid singularities (default 1e-5)
    expansion_order : number of terms in the multipole / local expansions,
                      capped at 5 (default 4)

    Returns
    -------
    acceleration : (N, 3) numpy array of gravitational accelerations
    """
    expansion_order = min(int(expansion_order), 5)

    positions = system.positions
    N         = system.N

    # ------------------------------------------------------------------
    # 1. Build oct-tree
    # ------------------------------------------------------------------
    minimum_position = np.min(positions, axis=0)
    maximum_position = np.max(positions, axis=0)
    centre           = 0.5 * (minimum_position + maximum_position)
    size             = np.max(maximum_position - minimum_position)

    root = fast_multipole.OctTreeNode(centre, size / 2.0, expansion_order)
    root.depth = 0

    for particle_index in range(N):
        fast_multipole.insert(root, system, particle_index)

    # ------------------------------------------------------------------
    # 2. Upward pass: compute mass distribution + multipole expansions
    # ------------------------------------------------------------------
    fast_multipole.compute_mass_distribution(root, system)
    fast_multipole.upward_pass(root, system)

    # ------------------------------------------------------------------
    # 3. Build interaction lists (structural, true O(N) FMM criterion)
    # ------------------------------------------------------------------
    fast_multipole.build_interaction_lists(root)

    # ------------------------------------------------------------------
    # 4. Downward pass: M2L translations + L2L propagation
    # ------------------------------------------------------------------
    fast_multipole.downward_pass(root)

    # ------------------------------------------------------------------
    # 5. Leaf evaluation: L2P (far-field) + P2P (near-field)
    # ------------------------------------------------------------------
    acceleration = fast_multipole.evaluate_leaves(root, system, G, softening)

    return acceleration
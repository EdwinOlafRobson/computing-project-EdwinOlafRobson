import numpy as np
from itertools import product as itertools_product


# =============================================================================
#  Oct-tree node  (extends the Barnes-Hut node with multipole + local arrays)
# =============================================================================

class OctTreeNode:
    def __init__(self, centre, half_size, expansion_order):
        self.centre       = np.asarray(centre, dtype=float)
        self.half_size    = float(half_size)
        self.expansion_order = expansion_order

        # Tree topology
        self.children       = [None] * 8
        self.parent         = None
        self.particle_index = None          # non-None only in a leaf with 1 particle

        # Mass / centre-of-mass (all masses == 1)
        self.mass            = 0.0
        self.centre_of_mass  = np.zeros(3)

        # FMM expansions stored as flat complex arrays indexed by (l, m)
        # with m in [-l, l].  Total coefficients = (p+1)^2.
        num_coefficients = (expansion_order + 1) ** 2
        self.multipole_expansion = np.zeros(num_coefficients, dtype=complex)
        self.local_expansion     = np.zeros(num_coefficients, dtype=complex)

        # Tree-level (root = 0) assigned during build
        self.depth = 0


# =============================================================================
#  Reused oct-tree helpers (identical logic to Barnes-Hut)
# =============================================================================

def get_octant(node, position):
    """Return the octant index (0-7) encoding which sub-cube *position* falls in.
    Bits are arranged as {z y x} so bit-0 → x, bit-1 → y, bit-2 → z."""
    index = 0
    if position[0] > node.centre[0]:
        index |= 1
    if position[1] > node.centre[1]:
        index |= 2
    if position[2] > node.centre[2]:
        index |= 4
    return index


def create_child(node, octant):
    """Create the child OctTreeNode for *octant* of *node*."""
    offset = np.array([
        0.5 if octant & 1 else -0.5,
        0.5 if octant & 2 else -0.5,
        0.5 if octant & 4 else -0.5,
    ])
    new_centre = node.centre + offset * node.half_size
    child = OctTreeNode(new_centre, node.half_size / 2, node.expansion_order)
    child.parent = node
    child.depth  = node.depth + 1
    return child


def insert(node, system, index):
    """Insert particle *index* into the oct-tree rooted at *node*."""
    particle_position = system.positions[index]

    # Empty leaf → store particle here
    if node.particle_index is None and node.children[0] is None:
        node.particle_index = index
        return

    # Leaf that already holds a particle → subdivide first
    if node.children[0] is None:
        for octant in range(8):
            node.children[octant] = create_child(node, octant)

        existing_index = node.particle_index
        node.particle_index = None
        insert(node, system, existing_index)

    octant = get_octant(node, particle_position)
    insert(node.children[octant], system, index)


# =============================================================================
#  Solid harmonic helpers
# =============================================================================

def _factorial(n):
    """Integer factorial via a small lookup (n <= 2*max_order is enough)."""
    result = 1
    for k in range(2, n + 1):
        result *= k
    return result


def _associated_legendre(l_max, x):
    """Return a 2-D array P[l, m] of associated Legendre values P_l^m(x)
    for 0 <= l <= l_max and 0 <= m <= l, evaluated at *x* (cos θ).
    Uses the standard upward recurrence; no Condon-Shortley phase."""
    P = np.zeros((l_max + 1, l_max + 1))
    P[0, 0] = 1.0
    sin_theta = np.sqrt(max(1.0 - x * x, 0.0))
    for l in range(1, l_max + 1):
        # Diagonal recurrence  P_l^l
        P[l, l] = -(2 * l - 1) * sin_theta * P[l - 1, l - 1]
        # Sub-diagonal  P_l^{l-1}
        P[l, l - 1] = x * (2 * l - 1) * P[l - 1, l - 1]
        # Remaining entries via standard 3-term recurrence
        for m in range(l - 2, -1, -1):
            P[l, m] = (x * (2 * l - 1) * P[l - 1, m] - (l + m - 1) * P[l - 2, m]) / (l - m)
    return P


def _lm_index(l, m):
    """Flat index for the (l, m) coefficient, m in [-l, l].
    Layout: l=0 → index 0; l=1 → indices 1,2,3; etc.
    For m >= 0 the index is  l^2 + l + m."""
    return l * l + l + m


def regular_solid_harmonic(l_max, position):
    """Compute the regular solid harmonics R_l^m(r) for all l, m up to *l_max*.
    R_l^m = r^l / (l+|m|)! * Y_l^m(theta, phi)  (unnormalised convention).
    Returns a flat complex array indexed by _lm_index(l, m)."""
    num_coefficients = (l_max + 1) ** 2
    R = np.zeros(num_coefficients, dtype=complex)

    x, y, z = position
    r_sq  = x * x + y * y + z * z
    r     = np.sqrt(r_sq)
    r_xy  = np.sqrt(x * x + y * y)

    cos_theta = z / r       if r     > 1e-30 else 0.0
    phi       = np.arctan2(y, x)

    P = _associated_legendre(l_max, cos_theta)

    for l in range(l_max + 1):
        r_power = r ** l
        for m in range(-l, l + 1):
            abs_m       = abs(m)
            norm_factor = r_power / _factorial(l + abs_m)
            legendre    = P[l, abs_m]
            phase       = np.exp(1j * m * phi)
            R[_lm_index(l, m)] = norm_factor * legendre * phase

    return R


def irregular_solid_harmonic(l_max, position):
    """Compute the irregular solid harmonics I_l^m(r) for all l, m up to *l_max*.
    I_l^m = (l-|m|)! / r^{l+1} * Y_l^m(theta, phi).
    Returns a flat complex array indexed by _lm_index(l, m)."""
    num_coefficients = (l_max + 1) ** 2
    I_harmonics = np.zeros(num_coefficients, dtype=complex)

    x, y, z = position
    r_sq  = x * x + y * y + z * z
    r     = np.sqrt(r_sq)

    if r < 1e-30:
        return I_harmonics          # singular at origin → return zeros

    cos_theta = z / r
    phi       = np.arctan2(y, x)

    P = _associated_legendre(l_max, cos_theta)

    for l in range(l_max + 1):
        inv_r_power = 1.0 / (r ** (l + 1))
        for m in range(-l, l + 1):
            abs_m       = abs(m)
            norm_factor = _factorial(l - abs_m) * inv_r_power
            legendre    = P[l, abs_m]
            phase       = np.exp(1j * m * phi)
            I_harmonics[_lm_index(l, m)] = norm_factor * legendre * phase

    return I_harmonics


# =============================================================================
#  Upward pass  (P2M at leaves, M2M towards root)
# =============================================================================

def compute_mass_distribution(node, system):
    """Populate mass and centre_of_mass for every node bottom-up.
    All particles have mass 1 (mirrors Barnes-Hut)."""
    if node is None:
        return 0.0, np.zeros(3)

    if node.particle_index is not None:
        node.mass           = 1.0
        node.centre_of_mass = system.positions[node.particle_index].copy()
        return node.mass, node.centre_of_mass

    total_mass     = 0.0
    centre_of_mass = np.zeros(3)
    for child in node.children:
        if child is None:
            continue
        child_mass, child_com = compute_mass_distribution(child, system)
        total_mass     += child_mass
        centre_of_mass += child_mass * child_com

    if total_mass > 0.0:
        centre_of_mass /= total_mass

    node.mass           = total_mass
    node.centre_of_mass = centre_of_mass
    return total_mass, centre_of_mass


def particle_to_multipole(node, system):
    """P2M: compute the multipole expansion of a leaf node about its centre."""
    expansion_order   = node.expansion_order
    particle_position = system.positions[node.particle_index]
    offset            = particle_position - node.centre    # r_particle - r_cell
    node.multipole_expansion += regular_solid_harmonic(expansion_order, offset)
    # mass == 1 so no scaling needed


def multipole_to_multipole(parent, child):
    """M2M: shift child multipole expansion to parent centre and accumulate."""
    expansion_order = parent.expansion_order
    shift           = child.centre - parent.centre          # child centre relative to parent centre

    regular_shift = regular_solid_harmonic(expansion_order, shift)

    for l in range(expansion_order + 1):
        for m in range(-l, l + 1):
            value = 0.0 + 0.0j
            for l_prime in range(l + 1):
                for m_prime in range(-l_prime, l_prime + 1):
                    l_double_prime = l - l_prime
                    m_double_prime = m - m_prime
                    if abs(m_double_prime) > l_double_prime:
                        continue
                    value += (child.multipole_expansion[_lm_index(l_prime, m_prime)]
                              * regular_shift[_lm_index(l_double_prime, m_double_prime)])
            parent.multipole_expansion[_lm_index(l, m)] += value


def upward_pass(node, system):
    """Recursively build multipole expansions from leaves to root."""
    if node is None:
        return

    if node.particle_index is not None:
        # Leaf with one particle
        particle_to_multipole(node, system)
        return

    for child in node.children:
        upward_pass(child, system)

    for child in node.children:
        if child is not None and child.mass > 0.0:
            multipole_to_multipole(node, child)


# =============================================================================
#  Interaction list construction
# =============================================================================

def _collect_nodes_at_depth(root, target_depth):
    """Return all nodes at exactly *target_depth* in the tree."""
    if root is None:
        return []
    if root.depth == target_depth:
        return [root]
    result = []
    for child in root.children:
        result.extend(_collect_nodes_at_depth(child, target_depth))
    return result


def _nodes_are_adjacent(node_a, node_b):
    """Two nodes (at the same depth, same half_size) are adjacent if their
    centres differ by at most 2 * half_size in every coordinate."""
    threshold = 2.0 * node_a.half_size + 1e-10
    return all(abs(node_a.centre[d] - node_b.centre[d]) <= threshold for d in range(3))


def build_interaction_lists(root):
    """Assign interaction_list to every node in the tree.

    A node *v* is in the interaction list of node *u* if:
      1. *u* and *v* are at the same depth.
      2. *u* and *v* are NOT adjacent (well-separated).
      3. Their parents ARE adjacent (so they were not separated at the level above).
    """
    max_depth = _tree_max_depth(root)

    for depth in range(2, max_depth + 1):
        nodes_at_depth = _collect_nodes_at_depth(root, depth)
        for node in nodes_at_depth:
            node.interaction_list = []

        for node in nodes_at_depth:
            if node.parent is None:
                node.interaction_list = []
                continue
            # Collect all children of node.parent's neighbours
            parent_neighbours = [
                other_node for other_node in _collect_nodes_at_depth(root, depth - 1)
                if _nodes_are_adjacent(node.parent, other_node)
            ]
            for parent_neighbour in parent_neighbours:
                for candidate in parent_neighbour.children:
                    if candidate is None:
                        continue
                    if candidate is node:
                        continue
                    if not _nodes_are_adjacent(node, candidate):
                        node.interaction_list.append(candidate)

    # Ensure root and depth-1 nodes have empty interaction lists
    for depth in range(min(2, max_depth + 1)):
        for node in _collect_nodes_at_depth(root, depth):
            node.interaction_list = []


def _tree_max_depth(node):
    if node is None:
        return 0
    if all(child is None for child in node.children):
        return node.depth
    return max(_tree_max_depth(child) for child in node.children)


# =============================================================================
#  Downward pass  (M2L for interaction lists, L2L towards leaves)
# =============================================================================

def multipole_to_local(target_node, source_node):
    """M2L: convert the multipole expansion of *source_node* into a local
    expansion contribution at *target_node*."""
    expansion_order  = target_node.expansion_order
    displacement     = target_node.centre - source_node.centre

    irregular_shifted = irregular_solid_harmonic(2 * expansion_order, displacement)

    for l in range(expansion_order + 1):
        for m in range(-l, l + 1):
            value = 0.0 + 0.0j
            for l_prime in range(expansion_order + 1):
                for m_prime in range(-l_prime, l_prime + 1):
                    l_combined = l + l_prime
                    m_combined = m + m_prime
                    if l_combined > 2 * expansion_order:
                        continue
                    if abs(m_combined) > l_combined:
                        continue
                    sign   = (-1) ** l_prime
                    value += (sign
                              * source_node.multipole_expansion[_lm_index(l_prime, m_prime)]
                              * irregular_shifted[_lm_index(l_combined, m_combined)])
            target_node.local_expansion[_lm_index(l, m)] += value


def local_to_local(parent, child):
    """L2L: shift the parent local expansion down to the child centre."""
    expansion_order = parent.expansion_order
    shift           = child.centre - parent.centre

    regular_shift = regular_solid_harmonic(expansion_order, shift)

    for l in range(expansion_order + 1):
        for m in range(-l, l + 1):
            value = 0.0 + 0.0j
            for l_prime in range(l, expansion_order + 1):
                for m_prime in range(-l_prime, l_prime + 1):
                    l_double_prime = l_prime - l
                    m_double_prime = m_prime - m
                    if abs(m_double_prime) > l_double_prime:
                        continue
                    value += (parent.local_expansion[_lm_index(l_prime, m_prime)]
                              * regular_shift[_lm_index(l_double_prime, m_double_prime)])
            child.local_expansion[_lm_index(l, m)] += value


def downward_pass(node):
    """Recursively propagate local expansions from root to leaves."""
    if node is None:
        return

    # M2L: accumulate contributions from this node's interaction list
    for source in getattr(node, 'interaction_list', []):
        if source.mass > 0.0:
            multipole_to_local(node, source)

    # L2L: push local expansion down to children
    for child in node.children:
        if child is not None:
            local_to_local(node, child)
            downward_pass(child)


# =============================================================================
#  Leaf evaluation  (L2P far-field + P2P direct near-field)
# =============================================================================

def local_to_particle(node, system, particle_index):
    """L2P: evaluate the gradient of the local expansion at *particle_index*'s
    position to obtain the smooth (far-field) acceleration contribution."""
    expansion_order   = node.expansion_order
    particle_position = system.positions[particle_index]
    offset            = particle_position - node.centre

    # We need the gradient of Re[ sum_{l,m} L_l^m * R_l^m(r) ].
    # Compute via finite differences on the regular solid harmonics — robust
    # and expansion-order agnostic for orders up to 5.
    delta = 1e-5
    acceleration = np.zeros(3)

    for dimension in range(3):
        offset_forward  = offset.copy(); offset_forward[dimension]  += delta
        offset_backward = offset.copy(); offset_backward[dimension] -= delta

        regular_forward  = regular_solid_harmonic(expansion_order, offset_forward)
        regular_backward = regular_solid_harmonic(expansion_order, offset_backward)

        potential_forward  = np.dot(node.local_expansion, regular_forward).real
        potential_backward = np.dot(node.local_expansion, regular_backward).real

        # gradient of potential → force per unit mass (mass == 1, G absorbed in driver)
        acceleration[dimension] = (potential_forward - potential_backward) / (2.0 * delta)

    return acceleration


def direct_particle_to_particle(positions_i, positions_j, index_i, G, softening):
    """Direct P2P summation: acceleration on particle *index_i* due to all
    particles in *positions_j* (excluding self-interaction)."""
    acceleration = np.zeros(3)
    position_i   = positions_i[index_i]

    for index_j in range(len(positions_j)):
        if index_j == index_i and np.array_equal(positions_i, positions_j):
            continue
        displacement = positions_j[index_j] - position_i
        distance_sq  = np.dot(displacement, displacement) + softening ** 2
        distance     = np.sqrt(distance_sq)
        acceleration += G * displacement / (distance_sq * distance)

    return acceleration


def _collect_leaf_nodes(node):
    """Return all leaf nodes (nodes that hold a particle) in the subtree."""
    if node is None:
        return []
    if node.particle_index is not None:
        return [node]
    leaves = []
    for child in node.children:
        leaves.extend(_collect_leaf_nodes(child))
    return leaves


def evaluate_leaves(root, system, G, softening):
    """For every leaf compute:
      - far-field acceleration via L2P from the local expansion
      - near-field acceleration via direct P2P with neighbouring leaves
    Returns a (N, 3) acceleration array."""
    N            = system.N
    acceleration = np.zeros((N, 3))

    all_leaves = _collect_leaf_nodes(root)

    # Build adjacency: two leaves are neighbours if they are adjacent cells
    for leaf in all_leaves:
        particle_index = leaf.particle_index

        # --- Far-field contribution from local expansion ---
        acceleration[particle_index] += local_to_particle(leaf, system, particle_index)

        # --- Near-field: direct summation with adjacent leaves (including self-leaf) ---
        for other_leaf in all_leaves:
            if _nodes_are_adjacent(leaf, other_leaf) or leaf is other_leaf:
                # Collect the single particle in other_leaf and sum directly
                other_index = other_leaf.particle_index
                if other_index == particle_index:
                    continue
                displacement = system.positions[other_index] - system.positions[particle_index]
                distance_sq  = np.dot(displacement, displacement) + softening ** 2
                distance     = np.sqrt(distance_sq)
                acceleration[particle_index] += G * displacement / (distance_sq * distance)

    return acceleration


# =============================================================================
#  Driver function
# =============================================================================

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

    root = OctTreeNode(centre, size / 2.0, expansion_order)
    root.depth = 0

    for particle_index in range(N):
        insert(root, system, particle_index)

    # ------------------------------------------------------------------
    # 2. Upward pass: compute mass distribution + multipole expansions
    # ------------------------------------------------------------------
    compute_mass_distribution(root, system)
    upward_pass(root, system)

    # ------------------------------------------------------------------
    # 3. Build interaction lists (structural, true O(N) FMM criterion)
    # ------------------------------------------------------------------
    build_interaction_lists(root)

    # ------------------------------------------------------------------
    # 4. Downward pass: M2L translations + L2L propagation
    # ------------------------------------------------------------------
    downward_pass(root)

    # ------------------------------------------------------------------
    # 5. Leaf evaluation: L2P (far-field) + P2P (near-field)
    # ------------------------------------------------------------------
    acceleration = evaluate_leaves(root, system, G, softening)

    return acceleration
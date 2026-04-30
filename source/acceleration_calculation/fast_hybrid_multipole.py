import numpy as np
 

class OctTreeNode:
    def __init__(self, centre, half_size):
        self.centre     = np.asarray(centre, dtype=float)
        self.half_size  = float(half_size)
 
        self.children       = [None] * 8
        self.parent         = None
        self.particle_index = None
 
        self.mass           = 0.0
        self.centre_of_mass = np.zeros(3)
        self.depth          = 0
 

        # Field from interior particles.
        # Second-order expansion, monopole dipole quarpole, 9 real coefficients
        # Stored as [q, dx, dy, dz, Qxx, Qyy, Qzz, Qxy, Qxz, Qyz]
        
        self.multipole = np.zeros(10)

        # Field from exterior particles.
        
        self.local = np.zeros(10)
 
        self.interaction_list = []
 
 
def get_octant(node, position):

    ''' Finds the octant of a particle. Stores position in bits as {xyz}. '''

    centre = node.centre
    index = 0
    if position[0] > centre[0]:
        index |= 1

    if position[1] > centre[1]: 
        index |= 2

    if position[2] > centre[2]:
        index |= 4

    return index
 

def create_child(node, octant):

    ''' Creates new node from old node.'''

    offset = np.array([
    0.5 if octant & 1 else -0.5,
    0.5 if octant & 2 else -0.5,
    0.5 if octant & 4 else -0.5
    ])

    child = OctTreeNode(node.centre + offset * node.half_size, node.half_size / 2)
    child.parent = node
    child.depth = node.depth + 1
    return child
 
 
def insert(node, system, index):
    particle_position = system.positions[index]

    # Empty leaf.

    if node.particle_index is None and node.children[0] is None:
        node.particle_index = index
        return

    if node.children[0] is None:
        for i in range(8):
            node.children[i] = create_child(node, i)

        # Puts any exising particle into a child node. This is the recursive part.

        if node.particle_index is not None:
            old_index = node.particle_index
            node.particle_index = None
            insert(node, system, old_index)

    octant = get_octant(node, particle_position)
    insert(node.children[octant], system, index)


 
def monopole_dipole_quadrupole(offset):

    """ Return the 10-element multipole contribution of a point mass of mass unity."""

    coefficients = np.zeros(10)
    x, y, z = offset

    coefficients[0] = 1
    coefficients[1] = x
    coefficients[2] = y
    coefficients[3] = z
    coefficients[4] = x * x
    coefficients[5] = y * y
    coefficients[6] = z * z
    coefficients[7] = x * y
    coefficients[8] = x * z
    coefficients[9] = y * z
    return coefficients
 
 
def translate_multipole(multipole, shift):
    
    """ Translates the mutlipole expansion by shift to be about a new centre. 
    Uses up to second order approximations."""

    shifted      = np.zeros(10)
    q            = multipole[0]
    dx, dy, dz   = multipole[1], multipole[2], multipole[3]
    sx, sy, sz   = shift
 
    
    shifted[0] = q
    
    shifted[1] = dx + q * sx
    shifted[2] = dy + q * sy
    shifted[3] = dz + q * sz
 
    # (r + s)^2 = r.r + 2r.s + s.s for quadropole
    shifted[4] = multipole[4] + 2 * dx * sx + q * sx * sx
    shifted[5] = multipole[5] + 2 * dy * sy + q * sy * sy
    shifted[6] = multipole[6] + 2 * dz * sz + q * sz * sz
    shifted[7] = multipole[7] + dx * sy + dy * sx + q * sx * sy
    shifted[8] = multipole[8] + dx * sz + dz * sx + q * sx * sz
    shifted[9] = multipole[9] + dy * sz + dz * sy + q * sy * sz
 
    return shifted
 
 
 
def upward_pass(node, system):
    if node is None:
        return
 
    if node.particle_index is not None:

        # If the leaf contains a particle do a particle to multipole claculation

        offset = system.positions[node.particle_index] - node.centre
        node.multipole = monopole_dipole_quadrupole(offset)
        node.mass = 1
        node.centre_of_mass = system.positions[node.particle_index].copy()
        return
 
    total_mass     = 0.0
    centre_of_mass_sum = np.zeros(3)
 
    for child in node.children:
        if child is None:
            continue
        upward_pass(child, system)
        if child.mass == 0.0:
            continue
 
        # Multipole to multipole shift and accumulation
        shift = child.centre - node.centre
        node.multipole += translate_multipole(child.multipole, shift)
        total_mass += child.mass
        centre_of_mass_sum += child.mass * child.centre_of_mass
 
    node.mass = total_mass
    if total_mass > 0.0:
        node.centre_of_mass = centre_of_mass_sum / total_mass
 
 

def collect_at_depth(node, target_depth):

    """ DFS search that returns how many nodes are at a given depth"""

    if node is None:
        return []
    if node.depth == target_depth:
        return [node]
    result = []
    for child in node.children:
        result.extend(collect_at_depth(child, target_depth))
    return result
 
 
def adjacent(node_a, node_b):

    ''' Checks if two cells are touching (within 1e-10)'''
    
    threshold = 2.0 * node_a.half_size + 1e-10
    return all(abs(node_a.centre[d] - node_b.centre[d]) <= threshold for d in range(3))
 
 
def max_depth(node):

    " Reterns the maximum depth of a node"

    if node is None:
        return 0
    if all(child is None for child in node.children):
        return node.depth
    return max(max_depth(child) for child in node.children)


 

def build_interaction_lists(root):

    """ Build interaction lists of nodes that interact via the multipole expansion.
    These nodes are non adjacent nodes, of the same depth, in adjacent parent nodes."""

    max_depth_val = max_depth(root)

    for depth in range(2, max_depth_val + 1):
        nodes = collect_at_depth(root, depth)

        for node in nodes:
            node.interaction_list = []

        parent_nodes = collect_at_depth(root, depth - 1)

        for node in nodes:
            if node.parent is None:
                continue

            # finds neighbouring parents
            parent_neighbours = [
                other for other in parent_nodes
                if adjacent(node.parent, other)
            ]

            # check children of neighbouring parents
            for parent_neighbour in parent_neighbours:
                for candidate in parent_neighbour.children:
                    if candidate is None or candidate is node:
                        continue
                    if not adjacent(node, candidate):
                        node.interaction_list.append(candidate)
 

def multipole_to_local(target_node, source_node):

    """ Converts the second order multipole of the source node into a taylor expansion about
    the centre of the target node."""

    # $\frac{1}{|\vec{R}-\vec{r'}|} = \sum_{n=0}^{2} \frac{1}{n!} (\vec{r'} \dot \nabla)^{n} \left [\frac{1}{|\vec{R}|} \right ], \vec{R} = \vec{R_{t}} - \vec{R_{s}}$

    displacement = target_node.centre - source_node.centre
    rx, ry, rz = displacement
    r_sq = rx*rx + ry*ry + rz*rz
    r = np.sqrt(r_sq)
 
    if r < 1e-30:
        return
 
    q = source_node.multipole[0]
    dipole = source_node.multipole[1:4]
    Qxx, Qyy, Qzz = source_node.multipole[4], source_node.multipole[5], source_node.multipole[6]
    Qxy, Qxz, Qyz = source_node.multipole[7], source_node.multipole[8], source_node.multipole[9]
 
    r3 = r_sq * r
    r5 = r3 * r_sq
    r7 = r5 * r_sq
 
    # Monopole contribution to local field (potential and gradient)
    # Phi_mono = q / r,  gradient = -q * r_vec / r^3
    target_node.local[0] +=  q / r
    target_node.local[1] += -q * rx / r3
    target_node.local[2] += -q * ry / r3
    target_node.local[3] += -q * rz / r3
 
    # Dipole Phi_dip = (d.r) / r^3
    dot_d_r = dipole[0]*rx + dipole[1]*ry + dipole[2]*rz
    target_node.local[0] += dot_d_r / r3
 
    # Gradient of dipole potential to get force
    for axis, r_axis in enumerate((rx, ry, rz)):
        target_node.local[1 + axis] += (
            dipole[axis] / r3
            - 3.0 * dot_d_r * r_axis / r5
        )
 
    # Quadrupole contribution:  Phi_quad = Q_ij r_i r_j / (2 r^5)
    quad_contraction = (Qxx*rx*rx + Qyy*ry*ry + Qzz*rz*rz
                        + 2.0*(Qxy*rx*ry + Qxz*rx*rz + Qyz*ry*rz))
    target_node.local[0] += 0.5 * quad_contraction / r5
 
    # Gradient of quadrupole potential to get force
    Q_dot_r = np.array([
        Qxx*rx + Qxy*ry + Qxz*rz,
        Qxy*rx + Qyy*ry + Qyz*rz,
        Qxz*rx + Qyz*ry + Qzz*rz,
    ])
    for axis, r_axis in enumerate((rx, ry, rz)):
        target_node.local[1 + axis] += (
            Q_dot_r[axis] / r5
            - 2.5 * quad_contraction * r_axis / r7
        )
 
 

def shift_local(local, shift):
  
    """ Shifts the far field Taylor expansion in the parent to be in the clid node."""

    shifted    = local.copy()
    sx, sy, sz = shift
    # The potential at the new origin picks up the gradient dotted with shift
    shifted[0] += local[1]*sx + local[2]*sy + local[3]*sz
    return shifted
 
 
def downward_pass(node):

    """ Far field expansions from the parent is moved down into the child."""
    # Child experiences forces from nodes further away than to be considered in its own expansion.

    if node is None:
        return
 
    for source in node.interaction_list:
        if source.mass > 0.0:
            multipole_to_local(node, source)
 
    for child in node.children:
        if child is not None:
            shift = child.centre - node.centre
            child.local += shift_local(node.local, shift)
            downward_pass(child)
 
 

def collect_leaves(node):

    """ List of all leaf nodes with partices rooted at the input node."""

    if node is None:
        return []
    if node.particle_index is not None:
        return [node]
    leaves = []
    for child in node.children:
        leaves.extend(collect_leaves(child))
    return leaves
 
 
def evaluate_leaves(root, system, G, softening):

    """ Applies far field to leaves and neaf field adjacent leave corrections"""

    acceleration = np.zeros((system.N, 3))
    all_leaves = collect_leaves(root)
 
    for leaf in all_leaves:
        particle_index = leaf.particle_index
        particle_position = system.positions[particle_index]

 
        # Local to particle
        acceleration[particle_index] += G * leaf.local[1:4]
 
        # Particle to Particle expansion
        for other_leaf in all_leaves:
            if other_leaf is leaf or not adjacent(leaf, other_leaf):
                continue
            other_position = system.positions[other_leaf.particle_index]
            displacement = other_position - particle_position
            distance_sq = np.dot(displacement, displacement) + softening ** 2
            distance = np.sqrt(distance_sq)
            acceleration[particle_index] += G * displacement / (distance_sq * distance)
 
    return acceleration

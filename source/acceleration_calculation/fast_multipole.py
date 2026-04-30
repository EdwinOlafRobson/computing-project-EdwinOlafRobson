import numpy as np

class NBodySystem:
    def __init__(self, positions, velocities):

        ''' Positions and velocities are N by 3 numpy arrays.'''

        if positions.shape[0] != velocities.shape[0]:
            raise ValueError(
                f"\n*** Different numbers of particle for position and velocity***")
        
        if positions.shape[1] != 3 or velocities.shape[1] != 3:
            raise ValueError(f"\n*** Positions or velocities are not 3 dimensional***")

        self.positions = np.array(positions, dtype=float)      
        self.velocities = np.array(velocities, dtype=float)  
        self.N = len(self.positions)



class OctTreeNode:
    def __init__(self, centre, half_size):
        self.centre     = np.asarray(centre, dtype=float)
        self.half_size  = float(half_size)
 
        self.children = [] 
        self.siblings = []
        self.cousins = []
        self.parent = None
        self.particles = []
        self.depth = 0
        self.interaction_list = []

        # Field from interior particles.
        # Second-order expansion, monopole dipole quarpole, 9 real coefficients
        # Stored as [q, dx, dy, dz, Qxx, Qyy, Qzz, Qxy, Qxz, Qyz]
        
        self.multipole = np.zeros(10)

        # Field from exterior particles.
        
        self.local = np.zeros(10)


def adjacent(node_a, node_b):
    '''
    Checks if two cells are touching (within 1e-10)
    '''
    threshold = 2.0 * node_a.half_size + 1e-10
    return all(abs(node_a.centre[d] - node_b.centre[d]) <= threshold for d in range(3))


def cousins(node):
    '''
    Fills in the cousin list for a node
    '''
    node.cousins = []
    parent = node.parent

    if parent is None:
        return
    

    for pibling in parent.siblings:
        for child in pibling.children:
            node.cousins.append(child)


def interaction_list(node):
    '''
    Fills in the interacton list from non-adjacent cousins
    '''
    for candidate in node.cousins:
        if adjacent(node, candidate) == False:
            node.interaction_list.append(candidate)


def contains(node, position):
    '''
    Checks if a particle is inside a node
    '''
    return all(
        abs(position[d] - node.centre[d]) <= node.half_size
        for d in range(3)
    )


def insert(particle_positions, index, node):
    '''
    Places particles into the node, recuring until it reaches a leaf
    '''
    if contains(node, particle_positions[index]) == False:
        return

    if len(node.children) == 0:
        node.particles.append(index)
        return

    for child in node.children:
        if contains(child, particle_positions[index]):
            insert(particle_positions, index, child)


def create_child(node):
    ''' 
    Creates 8 children from a node
    '''
    child_half_size = node.half_size / 2

    for i in range(8):

        offset = np.array([
        0.5 if i & 1 else -0.5,
        0.5 if i & 2 else -0.5,
        0.5 if i & 4 else -0.5
        ])

        child_centre = node.centre + offset * node.half_size
        child = OctTreeNode(child_centre, child_half_size)
        
        child.parent = node
        child.depth = node.depth + 1

        node.children.append(child)

    for child in node.children:
        child.siblings = [c for c in node.children if c is not child]
    

def tree(root_node, max_depth):
    '''
    Builds the Tree
    '''
    def recurse_create_child(node):


        if node.depth == max_depth:
            return

        create_child(node)

        for child in node.children:
            recurse_create_child(child)

    recurse_create_child(root_node)


def populate_tree(root_node, max_depth, particle_positions, particle_number):
    '''
    Inserts particles into the proper nodes, and fills in the node cousins and interaction list
    '''

    def recurse_cousins(node):

        if node.depth == max_depth:
            return
        
        cousins(node)

        # Recurse
        for child in node.children:
            recurse_cousins(child)

    recurse_cousins(root_node)

    def recurse_interaction_list(node):

        if node.depth == max_depth:
            return
        
        interaction_list(node)

        # Recurse
        for child in node.children:
            recurse_interaction_list(child)

    recurse_interaction_list(root_node)

    for i in range(particle_number):
        insert(particle_positions, i, root_node)



def compute_multipole_expansions(leaf_node, particle_positions):
    '''
    Computes 2nd order expansion of the potentisl field inside a leaf about its centre 
    due to the particles 
    '''
    q = 0.0
    p = np.zeros(3)
    Q = np.zeros((3, 3))

    for i in leaf_node.particles:
        r = particle_positions[i] - leaf_node.centre

        q +=1
        p += r
        Q +=  (3.0 * np.outer(r, r) - (np.dot(r, r) * np.eye(3)))

    leaf_node.multipole[0] = q
    leaf_node.multipole[1:4] = p
    leaf_node.multipole[4] = Q[0, 0]
    leaf_node.multipole[5] = Q[1, 1]
    leaf_node.multipole[6] = Q[2, 2]
    leaf_node.multipole[7] = Q[0, 1]
    leaf_node.multipole[8] = Q[0, 2]
    leaf_node.multipole[9] = Q[1, 2]



def compute_leaf_multipoles(root_node, max_depth, particle_positions):
    '''
    Goes through the tree and finds the multipole expansion of every single node
    '''
    def recurse(node):
        if node.depth < max_depth:
            for child in node.children:
                recurse(child)
            return

        compute_multipole_expansions(node, particle_positions)

    recurse(root_node)


def shift_multipole(source_node, target_node):
    '''
    Shifts multipole expansion from source_node centre to target_node centre.
    '''

   
    d = source_node.centre - target_node.centre

    q = source_node.multipole[0]
    p = source_node.multipole[1:4]
    Q = np.array([
        [source_node.multipole[4], source_node.multipole[7], source_node.multipole[8]],
        [source_node.multipole[7], source_node.multipole[5], source_node.multipole[9]],
        [source_node.multipole[8], source_node.multipole[9], source_node.multipole[6]]
    ])

  

    q_new = q

    p_new = p + q * d

    d_outer_d = np.outer(d, d)
    d_outer_p = np.outer(d, p)
    p_outer_d = np.outer(p, d)

    Q_new = (
        Q
        + 3.0 * (d_outer_p + p_outer_d)
        + 3.0 * q * d_outer_d
        - q * np.dot(d, d) * np.eye(3)
    )


    target_node.multipole[0] += q_new
    target_node.multipole[1:4] += p_new

    target_node.multipole[4] += Q_new[0, 0]
    target_node.multipole[5] += Q_new[1, 1]
    target_node.multipole[6] += Q_new[2, 2]
    target_node.multipole[7] += Q_new[0, 1]
    target_node.multipole[8] += Q_new[0, 2]
    target_node.multipole[9] += Q_new[1, 2]




def upwards_path(root_node, max_depth, particle_positions):
    '''
    Construct multipole expansion for leaf nodes and propagates them upwards 
    to the root. 
    '''
    compute_leaf_multipoles(root_node, max_depth, particle_positions)

    def recurse(node):

        if node.depth == max_depth:
            return

        # First recurse to children
        for child in node.children:
            recurse(child)

        node.multipole[:] = 0.0

        for child in node.children:
            shift_multipole(child, node)

    recurse(root_node)


def shift_multipole_to_local(source_node, target_node):
    '''
    Converts source multipole expansion into a local expansion at target node.
    (M2L operator, up to quadrupole order)
    '''

    R = target_node.centre - source_node.centre
    r = np.linalg.norm(R)

    q = source_node.multipole[0]
    p = source_node.multipole[1:4]
    Q = np.array([
        [source_node.multipole[4], source_node.multipole[7], source_node.multipole[8]],
        [source_node.multipole[7], source_node.multipole[5], source_node.multipole[9]],
        [source_node.multipole[8], source_node.multipole[9], source_node.multipole[6]]
    ])

    I = np.eye(3)

    # Precompute powers
    r2 = r * r
    r3 = r2 * r
    r5 = r3 * r2
    r7 = r5 * r2


    phi = (
        q / r
        + np.dot(p, R) / r3
        + 0.5 * np.sum(Q * (3 * np.outer(R, R) - r2 * I)) / r5
    )


    grad = (
        -q * R / r3
        + (p / r3 - 3 * np.dot(p, R) * R / r5)
        + (np.dot(Q,R) / r5 - 5 * (R @ Q @ R) * R / r7)
    )

    term1 = q * (3 * np.outer(R, R) - r2*I)/r5

    term2 = (
        3 * (np.outer(p,R) + np.outer(R,p))/r5
        - 15 * np.dot(p,R) * np.outer(R,R)/r7
    )

    term3 = (
        Q / r5
        - 5 * (np.outer(np.dot(Q, R), R) + np.outer(R, np.dot(Q, R))) / r7
    )

    H = term1 + term2 + term3


    target_node.local[0] += phi
    target_node.local[1:4] += grad

    target_node.local[4] += H[0, 0]
    target_node.local[5] += H[1, 1]
    target_node.local[6] += H[2, 2]
    target_node.local[7] += H[0, 1]
    target_node.local[8] += H[0, 2]
    target_node.local[9] += H[1, 2]


def shift_local(source_node, target_node):
    '''
    Shifts local expansion from source_node (parent) centre to target_node (child) centre.
    (L2L operator, up to quadrupole order)
    '''
    d = target_node.centre - source_node.centre

    phi  = source_node.local[0]
    grad = source_node.local[1:4]
    H    = np.array([
        [source_node.local[4], source_node.local[7], source_node.local[8]],
        [source_node.local[7], source_node.local[5], source_node.local[9]],
        [source_node.local[8], source_node.local[9], source_node.local[6]]
    ])

  
    phi_new  = phi + np.dot(grad, d) + 0.5 * d @ H @ d


    grad_new = grad + H @ d

    H_new = H.copy()

    target_node.local[0] += phi_new
    target_node.local[1:4] += grad_new
    target_node.local[4] += H_new[0, 0]
    target_node.local[5] += H_new[1, 1]
    target_node.local[6] += H_new[2, 2]
    target_node.local[7] += H_new[0, 1]
    target_node.local[8] += H_new[0, 2]
    target_node.local[9] += H_new[1, 2]

def evaluate_local_expansion(leaf_node, particle_position):
    '''
    Returns the acceleration that a unit mass expriances at a point due to the potential
    '''

    r = particle_position - leaf_node.centre

    phi = leaf_node.local[0]
    g = leaf_node.local[1:4]

    H = np.array([
        [leaf_node.local[4], leaf_node.local[7], leaf_node.local[8]],
        [leaf_node.local[7], leaf_node.local[5], leaf_node.local[9]],
        [leaf_node.local[8], leaf_node.local[9], leaf_node.local[6]]
    ])

    grad = g + H @ r
    acceleration = -grad

    return acceleration


def downward_pass(root_node, max_depth, particle_positions):
    '''
    Returns the acclereations of all the particles, found by evaluating the local 
    expansions of the leaf nodes at the particle positions, then performing particle-particle
    calculation between cousin nodes
    '''
    particle_accelerations = np.zeros_like(particle_positions)

    def recurse(node):

        
        for source in node.interaction_list:
            shift_multipole_to_local(source, node)

        if len(node.children) != 0:
            for child in node.children:
                shift_local(node, child)
                recurse(child)

        point_to_point_particles = []
        point_to_point_particles.extend(node.particles)

        for sibling in node.siblings:
            
            point_to_point_particles.extend(sibling.particles)
        
        # Point to point over adjacent cousins
        adjacent_cousins = [x for x in node.cousins if x not in node.interaction_list]
        for cousin in adjacent_cousins:
            point_to_point_particles.extend(cousin.particles)

        
        for particle_index in node.particles:
            
            local_acceleration = evaluate_local_expansion(node, particle_positions[particle_index])
            particle_accelerations[particle_index] += local_acceleration

            point_to_point_acceleration = np.zeros(3)
            
            for j in point_to_point_particles:
                if j != particle_index:

                    displacement = particle_positions[particle_index] - particle_positions[j]
                    abs_displacement = np.linalg.norm(displacement)
                    normalised_displacement = displacement / abs_displacement

                    acc = -normalised_displacement/ (abs_displacement ** 2)

                    point_to_point_acceleration += acc

            particle_accelerations[particle_index] += point_to_point_acceleration


    recurse(root_node)

    return particle_accelerations
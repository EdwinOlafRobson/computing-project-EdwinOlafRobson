import numpy as np

class OctTreeNode:
    def __init__(self, centre, half_size):
        self.centre = centre
        self.half_size = half_size

        self.children = [None for i in range(8)]
        self.particle_index = None

        self.mass = 0
        self.centre_of_mass = np.zeros(3)
        


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

    new_centre = node.centre + (offset * node.half_size)
    return OctTreeNode(new_centre, node.half_size/2)
    

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


def compute_mass_distribution(node, system):

    ''' Finds mass, centre of mass of a node. '''

    # Empty node.
    if node is None:
        return 0.0, np.zeros(3)
    
    # Leaf node with particle.
    if node.particle_index is not None:
        node.mass = 1.0
        node.centre_of_mass = system.positions[node.particle_index]
        return node.mass, node.centre_of_mass
    
    # Internal node
    total_mass = 0.0
    com = np.zeros(3)

    for child in node.children:
        if child is None:
            continue
        m, c = compute_mass_distribution(child, system)
        total_mass += m
        com += m * c

    if total_mass > 0:
        com /= total_mass

    node.mass = total_mass
    node.centre_of_mass = com

    return total_mass, com
    

def compute_gravitational_acceleration(node, system, i, theta, G, softening):

    ''' Uses Barnes-Hut to find the gravitational acceleration on particle i due to a node. '''

    if node is None or node.mass == 0:
        return np.zeros(3)

    pos_i = system.positions[i]

   
    if node.particle_index == i:
        return np.zeros(3)

    r = node.centre_of_mass - pos_i
    dist_sq = np.dot(r, r) + softening**2
    dist = np.sqrt(dist_sq)

    s = node.half_size * 2

    
    if (node.particle_index is not None) or (s / dist < theta):
        inv_dist3 = 1.0 / (dist_sq * dist)
        return G * r * inv_dist3

    
    acc = np.zeros(3)
    for child in node.children:
        if child is not None:
            acc += compute_gravitational_acceleration(child, system, i, theta, G, softening)

    return acc


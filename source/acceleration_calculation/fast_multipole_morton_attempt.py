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
 
        self.children       = [None] * 8
        self.parent         = None
        self.particle_index = None
        self.depth          = 0
        self.interaction_list = []


        # Field from interior particles.
        # Second-order expansion, monopole dipole quarpole, 9 real coefficients
        # Stored as [q, dx, dy, dz, Qxx, Qyy, Qzz, Qxy, Qxz, Qyz]
        
        self.multipole = np.zeros(10)

        # Field from exterior particles.
        
        self.local = np.zeros(10)
 



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


def _spread_bits(x):
    '''
    Truncates x into a 21-bit integer. Spreads this out into a 63 bit integer
    with every 3rd position filled
    Spread the bits of a 21-bit integer into every third bit position.
    e.g.  x = ...b2 b1 b0   →  ... b2 0 0 b1 0 0 b0
    '''
    # Truncates to 21 bit
    x &= 0x1FFFFF

    # Shifts the bits to be a 63 bit integer
    x = (x | (x << 32)) & 0x1F00000000FFFF
    x = (x | (x << 16)) & 0x1F0000FF0000FF
    x = (x | (x <<  8)) & 0x100F00F00F00F00F
    x = (x | (x <<  4)) & 0x10C30C30C30C30C3
    x = (x | (x <<  2)) & 0x1249249249249249
    return x


def compute_morton_codes(positions, depth):
    """
    Map each particle's 3-D position to a single 63-bit Morton (Z-order) code.
    Morton Code = ... z2 y2 xz z1 y1 x1 z0 y0 x0
    """
    mins  = positions.min(axis=0)
    maxs  = positions.max(axis=0)
    spans = maxs - mins
    norm  = (positions - mins) / spans

    # 2^depth - 1, the maximum grid index
    scale = (1 << depth) - 1 
    grid  = (norm * scale).astype(np.int64).clip(0, scale)

    x, y, z = grid[:, 0], grid[:, 1], grid[:, 2]

    codes = (
        _spread_bits(x) | (_spread_bits(y) << 1) | (_spread_bits(z) << 2) 
        )
    return codes




def radix_sort_indices(keys):
    '''
    Sort the array of 64-bit integer keys and return the permutation
    that puts them in ascending order along the Morton - Z curve.
    '''
    n      = len(keys)
    order  = np.arange(n, dtype=np.int64)
    keys   = keys.copy()

    BITS_PER_PASS = 8
    BUCKETS       = 1 << BITS_PER_PASS   # 256
    MASK          = BUCKETS - 1
    NUM_PASSES    = 64 // BITS_PER_PASS  # 8

    for p in range(NUM_PASSES):
        shift  = p * BITS_PER_PASS
        digits = (keys[order] >> shift) & MASK   # extract 8-bit digit

        # Count occurrences of each digit  →  O(N)
        counts = np.bincount(digits, minlength=BUCKETS)

        # Exclusive prefix sum gives the starting output position
        # for each bucket                  →  O(256) = O(1)
        starts = np.zeros(BUCKETS, dtype=np.int64)
        starts[1:] = np.cumsum(counts[:-1])

        # Scatter into new order array     →  O(N)
        new_order = np.empty(n, dtype=np.int64)
        for i in range(n):
            d            = digits[i]
            new_order[starts[d]] = order[i]
            starts[d]   += 1

        order = new_order

    return order




def _node_key(morton_code, level, max_depth):
    '''
    Each code is set out like z_d y_d x_d z_{d-1} y_{d-1} x_{d-1} ... z_0 y_0 x_0
    This function describes a node up to a set level of detail
    '''
    shift        = 3 * (max_depth - level)
    coarse_code  = int(morton_code) >> shift
    return (level, coarse_code)


def _children_keys(node_key, max_depth):
    '''
    Return the 8 child keys of a node (one level deeper).
    '''
    level, code = node_key
    child_level = level + 1
    base        = code << 3
    return [(child_level, base | octant) for octant in range(8)]


def _parent_key(node_key):
    level, code = node_key
    return (level - 1, code >> 3)



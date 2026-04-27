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



def simulate(system,
            step_function,
            acceleration_function, 
            steps,
            dt=0.001,
            G=1, 
            softening=1e-5,
            **acceleration_kwargs):

    '''Simulates and records trajectory of system.'''

    trajectory_positions = np.zeros((steps, system.N, 3))
    trajectory_velocities = np.zeros((steps, system.N, 3))

    def acceleration_function_with_arguments(sys):
        return acceleration_function(sys, G, softening, **acceleration_kwargs)

    for t in range(steps):
        trajectory_positions[t] = system.positions
        trajectory_velocities[t] = system.velocities

        step_function(system, acceleration_function_with_arguments, dt)
    
    return trajectory_positions, trajectory_velocities

   
        
def cubic_lattice(N, box_size=1):
    
    ''' Creates a Cartesian lattice. '''

    n = int(round(N ** (1/3)))
    if n**3 != N:
        print("N must be cubic!")
        exit()
    
    x = np.linspace(0, box_size, n, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')

    positions = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
    return positions



# Gravitational simulations.


def gravitational_random_position_random_velocity(N, box_size=1, max_velocity=1):
    positions = np.random.uniform(0, box_size, (N,3))
    velocities = np.random.uniform(-max_velocity, max_velocity, (N, 3))
    return NBodySystem(positions, velocities)
    
    
def gravitational_random_position_stationary_velocity(N, box_size=1):
    positions = np.random.uniform(0, box_size, (N,3))
    velocities = np.full((N,3), 0)
    return NBodySystem(positions, velocities)


def gravitational_uniform_position_random_velocity(N, box_size=1, max_velocity=1):
    positions = cubic_lattice(N, box_size)
    velocities = np.random.uniform(-max_velocity, max_velocity, (N, 3))
    return NBodySystem(positions, velocities)


def gravitational_uniform_position_stationary_velocity(N, box_size=1):
    positions = cubic_lattice(N, box_size)
    velocities = np.full((N,3), 0)
    return NBodySystem(positions, velocities)


def gravitational_constant_random_position_no_net_velocity(N, box_size=1, max_velocity=1):
    ''' Creates a reproduceable random system with no net momentum'''

    seed = 420   
    rng = np.random.default_rng(seed)
    positions = rng.uniform(0, box_size, size=(N, 3))

    seed = 69    
    rng = np.random.default_rng(seed)
    velocities = rng.uniform(-max_velocity, max_velocity, size=(N, 3))
    velocities -= velocities.mean(axis=0)
    return NBodySystem(positions, velocities)
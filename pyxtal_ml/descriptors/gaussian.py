import numpy as np
from pymatgen.core.structure import Structure

############################# Auxiliary Functions #############################

def distance(arr):
    """
    L2 norm for cartesian coordinates
    """
    return ((arr[0] ** 2 + arr[1] ** 2 + arr[2] ** 2) ** 0.5)


############################## Cutoff Functional ##############################

"""
This script provides three cutoff functionals:
    1. Cosine
    2. Polynomial
    3. Hyperbolic Tangent

All cutoff functionals have an 'Rc' attribute which is the cutoff radius;
The Rc is used to calculate the neighborhood attribute. The functional will
return zero if the radius is beyond Rc.

This script is adopted from AMP:
    https://bitbucket.org/andrewpeterson/amp/src/2865e75a199a?at=master
"""


class Cosine(object):
    """
    Cutoff cosine functional suggested by Behler:
    Behler, J., & Parrinello, M. (2007). Generalized neural-network 
    representation of high-dimensional potential-energy surfaces. 
    Physical review letters, 98(14), 146401.
    (see eq. 3)
    
    Args:
        Rc(float): the cutoff radius.
    """
    def __init__(self, Rc):
        
        self.Rc = Rc
        
    def __call__(self, Rij):
        """
        Args:
            Rij(float): distance between pair atoms.
            
        Returns:
            The value (float) of the cutoff Cosine functional, will return zero
            if the radius is beyond the cutoff value.
        """
        if Rij > self.Rc:
            return 0.0
        else:
            return (0.5 * (np.cos(np.pi * Rij / self.Rc) + 1.))
        
    def derivative(self, Rij):
        """
        Calculate derivative (dF/dRij) of the Cosine functional with respect
        to Rij.
        
        Args:
            Rij(float): distance between pair atoms.
            
        Returns:
            The derivative (float) of the Cosine functional.
        """
        if Rij > self.Rc:
            return 0.0
        else:
            return (-0.5 * np.pi / self.Rc * np.sin(np.pi * Rij / self.Rc))
        
    def todict(self):
        return {'name': 'Cosine',
                'kwargs': {'Rc': self.Rc
                           }
                }
        
        
class Polynomial(object):
    """
    Polynomial functional suggested by Khorshidi and Peterson:
    Khorshidi, A., & Peterson, A. A. (2016).
    Amp: A modular approach to machine learning in atomistic simulations. 
    Computer Physics Communications, 207, 310-324.
    (see eq. 9)

    Args:
        gamma(float): the polynomial power.
        Rc(float): the cutoff radius.
    """
    def __init__(self, Rc, gamma=4):
        self.gamma = gamma
        self.Rc = Rc
        
    def __call__(self, Rij):
        """
        Args:
            Rij(float): distance between pair atoms.
            
        Returns:
            The value (float) of the cutoff functional.
        """
        if Rij > self.Rc:
            return 0.0
        else:
            value = 1. + self.gamma * (Rij / self.Rc) ** (self.gamma + 1) - \
                (self.gamma + 1) * (Rij / self.Rc) ** self.gamma
            return value
        
    def derivative(self, Rij):
        """
        Derivative (dF/dRij) of the Polynomial functional with respect to Rij.
        
        Args:
            Rij(float): distance between pair atoms.
            
        Returns:
            The derivative (float) of the cutoff functional.
        """
        if Rij > self.Rc:
            return 0.0
        else:
            ratio = Rij / self.Rc
            value = (self.gamma * (self.gamma + 1) / self.Rc) * \
                (ratio ** self.gamma - ratio ** (self.gamma - 1))
        return value
    
    def todict(self):
        return {'name': 'Polynomial',
                'kwargs': {'Rc': self.Rc,
                           'gamma': self.gamma
                           }
                }
                        

class TangentH(object):
    """
    Cutoff hyperbolic Tangent functional suggested by Behler:
    Behler, J. (2015). 
    Constructing high‐dimensional neural network potentials: A tutorial review. 
    International Journal of Quantum Chemistry, 115(16), 1032-1050.
    (see eq. 7)

    Args:
        Rc(float): the cutoff radius.
    """
    def __init__(self, Rc):
        
        self.Rc = Rc
        
    def __call__(self, Rij):
        """
        Args:
            Rij(float): distance between pair atoms.
            
        Returns:
            The value (float) of the cutoff hyperbolic tangent functional, 
            will return zero if the radius is beyond the cutoff value.
        """
        if Rij > self.Rc:
            return 0.0
        else:
            return ((np.tanh(1.0 - (Rij / self.Rc))) ** 3)
        
    def derivative(self, Rij):
        """
        Calculate derivative (dF/dRij) of the hyperbolic Tangent functional 
        with respect to Rij.
        
        Args:
            Rij(float): distance between pair atoms.
            
        Returns:
            The derivative (float) of the hyberbolic tangent functional.
        """
        if Rij > self.Rc:
            return 0.0
        else:
            return (-3.0 / self.Rc * ((np.tanh(1.0 - (Rij / self.Rc))) ** 2 - \
                     (np.tanh(1.0 - (Rij / self.Rc))) ** 4))
        
    def todict(self):
        return {'name': 'TanH',
                'kwargs': {'Rc': self.Rc
                           }
                }

############################# Symmetry Functions ##############################

def calculate_G2(crystal, eta, cutoff_f='Cosine', Rc=6.5, Rs=0.0): # How do you pick eta?
    """
    Calculate G2 symmetry function.
    G2 function is a better choice to describe the radial feature of atoms in
    a crystal structure within the cutoff radius.
    
    One can refer to equation 9 in:
    Behler, J. (2015). Constructing high‐dimensional neural network 
    potentials: A tutorial review. 
    International Journal of Quantum Chemistry, 115(16), 1032-1050.

    Parameters
    ----------
    crystal: object
        Pymatgen crystal structure object.
    eta: float
        The parameter of G2 symmetry function.
    cutoff_f: str
        Cutoff functional. Default is Cosine functional.
    Rc: float
        Cutoff radius which the symmetry function will be calculated.
        Default value is 6.5 as suggested by Behler.
    Rs: float
        Determine the shift from the center of the Gaussian.
        Default value is zero.

    Returns
    -------
    G2 : float
        G2 symmetry value.
    """
    # Cutoff functional
    if cutoff_f == 'Cosine':
        func = Cosine(Rc=Rc)
    elif cutoff_f == 'Polynomial':
        func = Polynomial(Rc=Rc)
    elif cutoff_f == 'TangentH':
        func = TangentH(Rc=Rc)
    else:
        raise NotImplementedError('Unknown cutoff functional: %s' %cutoff_f)
    
    # Get positions of core atoms
    n_core = crystal.num_sites
    n_core_cartesian = crystal.cart_coords
    n_core_positions = np.apply_along_axis(distance, 1, n_core_cartesian)
    
    # Their neighbors within the cutoff radius
    neighbors = crystal.get_all_neighbors(Rc)
    n_neighbors = len(neighbors[1])
    
    G2 = []
    for i in range(n_core):
        G2_core = 0
        for j in range(n_neighbors):
            Rij = np.linalg.norm(n_core_positions[i] - neighbors[i][j][1])
            G2_core += (np.exp(-eta * Rij ** 2.) / (Rc ** 2.) * 
                        func(Rij))
        G2.append(G2_core)
    
    return G2


crystal = Structure.from_file('POSCARs/POSCAR-NaCl')
x = calculate_G2(crystal, eta =2)
print(x)

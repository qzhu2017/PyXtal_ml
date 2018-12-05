"""
This is a cutoff script which contains several cutoff functions.

All cutoff functions need a "todict" method.

All cutoff functions also need to have an 'Rc' attribute which is the maximum
distance at which properties are calculated; this will be used in calculating
neighborlists.

This script is adopted from AMP:
    https://bitbucket.org/andrewpeterson/amp/src/2865e75a199a?at=master
"""

import numpy as np

def dict2cutoff(dct):
    """
    This function converts a dictionary(which was created with the to_dict)
    method of one of the cutoff classes) into an instantiated version of the
    class. Modeled after ASE's dict2constraint function.
    """
    if len(dct) != 2:
        raise RuntimeError('Cutoff dictionary must have only two values, '
                           ' "name" and "kwargs".')
    return globals()[dct['name']](**dct['kwargs'])

class Cosine(object):
    """
    Cosine functional suggested by Behler.
    
    Args:
        Rc(float): the maximum distance at which properties are calculated.
    """
    def __init__(self, Rc):
        
        self.Rc = Rc
        
    def __call__(self, Rij):
        """
        Args:
            Rij(float): distance between pair atoms.
            
        Returns:
            The value (float) of the cutoff function.
        """
        if Rij > self.Rc:
            return 0.
        else:
            return (0.5 * (np.cos(np.pi * Rij / self.Rc) + 1.))
        
    def derivative(self, Rij):
        """
        Derivative (dF/dRij) of the Cosine function with respect to Rij.
        
        Args:
            Rij(float): distance between pair atoms.
            
        Returns:
            The derivative (float) of the Cosine function.
        """
        if Rij > self.Rc:
            return 0.
        else:
            return (-0.5 * np.pi / self.Rc * np.sin(np.pi * Rij / self.Rc))
        
    def todict(self):
        return {'name': 'Cosine',
                'kwargs': {'Rc': self.Rc
                           }
                }
        
    def __repr(self):
        return ('<Cosine cutoff with Rc=%.3f from descriptor.cutoffs>'
                %self.Rc)
        
class Polynomial(object):
    """
    Polynomial functional suggested by Khorshidi and Peterson.
    
    Args:
        gamma(float): the power of polynomial.
        Rc(float): the maximum distance at which properties are calculated.
    """
    def __init__(self, Rc, gamma=4):
        self.gamma = gamma
        self.Rc = Rc
        
    def __call__(self, Rij):
        """
        Args:
            Rij(float): distance between pair atoms.
            
        Returns:
            The value (float) of the cutoff function.
        """
        if Rij > self.Rc:
            return 0.
        else:
            value = 1. + self.gamma * (Rij / self.Rc) ** (self.gamma + 1) - \
                (self.gamma + 1) * (Rij / self.Rc) ** self.gamma
            return value
        
    def derivative(self, Rij):
        """
        Derivative (dF/dRij) of the Polynomial function with respect to Rij.
        
        Args:
            Rij(float): distance between pair atoms.
            
        Returns:
            The derivative (float) of the cutoff function.
        """
        if Rij > self.Rc:
            return 0.
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
                
    def __repr__(self):
        return ('<Polynomial cutoff with Rc=%.3f and gamma=%i '
                'from descriptor.cutoffs>'
                %(self.Rc, self.gamma))
from __future__ import print_function, division
import os
import csv
import re
import json
import functools
import random
import warnings
import numpy as np
from pymatgen.core.structure import Structure

class GaussianDistance():
    """
    Expands the distance by Gaussian basis.
    
    Args:
        dmin (float): Min interatomic distance.
        dmax (float): Max interatomic distance.
        step (float): Step size for the Gaussian filter.
    """
    def __init__(self, dmin, dmax, step, var=None):
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax+step, step)
        if var is None:
            var = step
        self.var = var
        
    def expand(self, distances):
        """
        Apply Gaussian distance filter to a numpy distance array.
        
        Returns:
            Expanded distance matrix with the last dimension of length
            len(self.filter)
        """
        X = np.exp(-(distances[..., np.newaxis] - self.filter)**2 / 
                   self.var**2)
        return X
    
class ElementInitializer():
    """
    Initializing the vector representation for elements.
    
    Args:
        atom_types (int): types of elements in their perspective representation
                            (1-100; H-Fm).
    """
    def __init__(self, elem_types):
        self.elem_types = set(elem_types)
        self._embedding = {}
        
    def get_elem_fea(self, elem_type):
        assert elem_type in self.elem_types
        return self._embedding[elem_type]
    
class ElementJSONInitializer(ElementInitializer):
    """
    Storing information in elem_init.json to _embedding dictionary.
    
    Args:
        elem_embedding_file (string): provide a path to find the elem_init.json
                                        file.
    """
    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(key): value for key, value in
                          elem_embedding.items()}
        elem_types = set(elem_embedding.keys())
        super(ElementJSONInitializer, self).__init__(elem_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)
    
class crystal_graph():
    """
    Constructing crystal graph for a given crystal structure.
    
    Args:
        directory
    """
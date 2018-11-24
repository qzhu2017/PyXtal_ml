from __future__ import print_function, division
import os
import json
import warnings
import numpy as np
from optparse import OptionParser
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
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
    
class crystalgraph():
    """
    A class to compute the connectivity graph for a crystal, created based on
    https://github.com/txie-93/cgcnn/blob/master/cgcnn/data.py
    
    Args:
        crystal: crystal class from pymatgen
        jsonfile: atom_init.json file
        max_neighbor (int): the maximum number of neighbor for each atom.
        radius (float): the cutoff distance for Gaussian function.
        dmin: starting distance for Gaussian function.
        step(float): incremental in step for Gaussian function.
        
    --------------------------------------------------------------------------
    Remarks:
        Atom_fea is an encoded vector with binary numbers using 
        one hot encoding.
        For discrete values, the feature are encoded according to the
        category that the value belongs to; 
        for continuous values, the range of property values is
        evenly divided to 10 categories and the vectors are encoded 
        accordingly. The full list of atom and bond properties as well as 
        their ranges are in Table S2 and Table S3 of the PRL paper.
    --------------------------------------------------------------------------
    """
    def __init__(self, crystal, jsonfile='atom_init.json',
                 max_neighbor=12, radius=9, dmin=0, step=0.1):
        finder = SpacegroupAnalyzer(crystal, symprec=0.06,
                                    angle_tolerance=5)
        self.crystal = finder.get_conventional_standard_structure()
        self.directory = 'pyxtal_ml/descriptors/'
        self.max_neighbor = max_neighbor
        self.radius = radius
        self.elem_init_file = os.path.join(self.directory, jsonfile)
        self.gaussd = GaussianDistance(dmin=dmin,dmax=self.radius,step=step)
        
        # get element feature & neighbor feature
        self.get_elem_fea()
        self.get_neighbor_fea()
        
        # crystal graph for a structure
        self.crystal_graph = np.vstack((self.elem_fea, self.all_neighbor))
        
        
        
    def get_elem_fea(self):
        elem_fea_init = ElementJSONInitializer(self.elem_init_file)
        elem_fea = []
        for i in self.crystal.species:
            elem_fea.append(elem_fea_init.get_elem_fea(i.number))
        self.elem_fea = np.array(elem_fea)

        
    def get_neighbor_fea(self, crystal):
        """
        compute the crystal graph.
        Args:
            crystal: crystal structure information
        
        """
        all_neighbors = crystal.get_all_neighbors(self.radius,
                                                  include_index=True)
        all_neighbors = [sorted(neighbor, key=lambda x:x[1])
                        for neighbor in all_neighbors] # sort by distance
        
        neighbor_fea, neighbor_fea_site = [], []
        for neighbor in all_neighbors:
            if len(neighbor) < self.max_neighbor:
                warnings.warn('{} not find enough neigbors to build graph. '
                              'If this happens frequently, please consider '
                              'increasing the radius.'.format(
                                      self.cyrstal.composition.formula))
                neighbor_fea.append(list(map(lambda x: x[1],neighbor)) + 
                                    [self.radius + 1.] * 
                                    (self.max_neighbor - len(neighbor)))
                neighbor_fea_site.append(list(map(lambda x: x[2], neighbor)) +
                                         [0] * 
                                         (self.max_neighbor-len(neighbor)))
            else:
                neighbor_fea.append(list(map(lambda x: x[1], 
                                             neighbor[:self.max_neighbor])))
                neighbor_fea_site.append(list(map(lambda x: x[1],
                                                  neighbor[:self.max_neighbor])))
        
        # Reshaping neighbor_fea
        neighbor_fea = np.array(neighbor_fea)
        neighbor_fea = self.gaussd.expand(neighbor_fea)
        shape_nf = neighbor_fea.shape
        shape_nf_01 = shape_nf[0]*shape_nf[1]
        self.neighbor_fea = np.reshape(neighbor_fea,(shape_nf_01, shape_nf[3]))
        
        # Reshaping neighbor_fea_site
        neighbor_fea_site = np.array(neighbor_fea_site)
        self.neighbor_fea_site = np.ravel(neighbor_fea_site)
        
        # Put neighbor_fea & neighbor_fea_site together
        self.all_neighbor = np.column_stack((self.neighbor_fea,
                                             self.neighbor_fea_site))
        
        
if __name__ == '__main__':
    # ------------------------ Options -------------------------------------
    parser = OptionParser()
    parser.add_option("-c", "--crystal", dest="structure", default='',
                      help="crystal from file, cif or poscar, REQUIRED",
                      metavar="crystal")

    (options, args) = parser.parse_args()

    if options.structure.find('cif') > 0:
        fileformat = 'cif'
    else:
        fileformat = 'poscar'

    test = Structure.from_file(options.structure)
    cg = crystalgraph(test)
    print('\n atom_fea_id:\n', cg.elem_fea)
    print('\n nbr_fea:\n', cg.neighbor_fea)
    print('\n nbr_fea_id:\n', cg.neighbor_fea_site)
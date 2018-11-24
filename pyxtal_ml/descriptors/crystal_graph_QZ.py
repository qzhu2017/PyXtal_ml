import numpy as np
from pymatgen.core.structure import Structure
from optparse import OptionParser
import numpy as np
import warnings
import json

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
    a class to compute the connecvitiy graph for the crystal, created based on 
    https://github.com/txie-93/cgcnn/blob/master/cgcnn/data.py
    Parameters
    ----------
    max_nbr: int
        The maximum number of neighbors while constructing the crystal graph
    cutoff: float
        The cutoff radius for searching neighbors
    Returns
    -------
    atom_fea: atom feature, 1D array (ni)
    nbr_fea: neighbour feature, 2D array (n_i, max_nbr)
    nbr_fea_idx: 2D array, same size as in nbr_fea (n_i, max_nbr)
    -------------------------------------------------------------
    Remarks:
    Atom_fea is encoded a vector with binary numbers as follows,
    For discrete values, the vectors are encoded according to the
    category that the value belongs to; 
    for continuous values, the range of property values is
    evenly divided to 10 categories and the vectors are encoded accordingly. 
    The full list of atom and bond properties as well as their ranges are 
    in Table S2 and Table S3 of the PRL paper. 

    Nbr_fea: A step of Gaussian filter was proposed. 
    QZ: I am not sure if it is necessary.
    --------------------------------------------------------------

    """
    def __init__(self, struc, max_nbr=12, cutoff=8.0, jsonfile='atom_init.json'):
        self.max_nbr = max_nbr
        self.cutoff = cutoff
        self.crystal = struc
        self.jsonfile = jsonfile
        self.get_atom_fea()
        self.get_nbr_fea()

    def get_atom_fea(self):
        ari = ElementJSONInitializer(self.jsonfile)
        atom_fea = []
        for i in self.crystal.species:
            atom_fea.append(ari.get_elem_fea(i.number))
        self.atom_fea = np.array(atom_fea)

    def get_nbr_fea(self, expand=True):
        # a better way to get the table is via voronoi analysis!
        all_nbrs = self.crystal.get_all_neighbors(self.cutoff, include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]

        nbr_fea_idx, nbr_fea = [], []
        for nbr in all_nbrs:
            if len(nbr) < self.max_nbr:
                warnings.warn('{} not find enough neighbors to build graph. '
                              'If it happens frequently, consider increase '
                              'radius.'.format(cif_id))
                # pad 0 values
                nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) +
                                   [0] * (self.max_num_nbr - len(nbr)))
                nbr_fea.append(list(map(lambda x: x[1], nbr)) +
                               [self.radius + 1.] * (self.max_num_nbr -
                                                     len(nbr)))
            else:
                nbr_fea_idx.append(list(map(lambda x: x[2],
                                            nbr[:self.max_nbr])))
                nbr_fea.append(list(map(lambda x: x[1],
                                        nbr[:self.max_nbr])))
        self.nbr_fea_idx = np.array(nbr_fea_idx)
        self.nbr_fea = np.array(nbr_fea)


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
    print('\n atom_fea_id:\n', cg.atom_fea)
    print('\n nbr_fea:\n', cg.nbr_fea)
    print('\n nbr_fea_id:\n', cg.nbr_fea_idx)

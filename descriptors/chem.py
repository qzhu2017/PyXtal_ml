from monty.serialization import loadfn
from pymatgen.core.structure import Structure
import numpy as np
from optparse import OptionParser
import os.path as op

filename = op.join(op.dirname(__file__), 'Elements.json')
ele_data = loadfn(filename)


class Chem(object):
    """a class of crystal structure.
    Args:
        crystal: crystal class from pymatgen
    Attributes:
        crystal: crystal class
        mean_chem: chemical descriptor for the given structure
    """

    def __init__(self, struc):
        """Return a Chem object with the proper info"""
        comp = struc.composition
        el_dict = comp.get_el_amt_dict()
        arr = []
        for k, v in el_dict.items():
            des = self.get_descrp_arr(k)
            arr.append(des)
        self.mean_chem = np.mean(arr, axis=0)

    def get_descrp_arr(self, elm):
        # the current json file has only 82 elements, some are missing
        if elm in ['Pa', 'Ac', 'Pu', 'Np', 'Am', 'Bk', 'Cf', 'Cm', 'Es',
                   'Fm', 'Lr', 'Md', 'No']:
            elm = 'Th'
        elif elm in ['Eu', 'Pm']:
            elm = 'La'
        elif elm in ['Xe', 'Rn']:
            elm = 'Kr'
        elif elm in ['At']:
            elm = 'I'
        elif elm in ['Fr']:
            elm = 'Cs'
        elif elm in ['Ra']:
            elm = 'Ba'
        d = ele_data[elm]
        arr = []
        for k, v in d.items():
            arr.append(v)
        arr = np.array(arr).astype(float)
        return arr


if __name__ == "__main__":
    # -------------------------------- Options -------------------------
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
    chem = Chem(test)
    print(chem.mean_chem)
    print('shape of this descriptor: ', np.shape(chem.mean_chem))

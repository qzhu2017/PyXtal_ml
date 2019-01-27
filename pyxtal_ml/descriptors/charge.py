from pymatgen.core.structure import Structure
import numpy as np
from optparse import OptionParser
from pyxtal_ml.descriptors.stats import descriptor_stats
import os.path as op
from monty.serialization import loadfn

filename = op.join(op.dirname(__file__), 'element_charge.json')
ele_data = loadfn(filename)


class Charge(object):
    '''
    '''

    def __init__(self, struc):
        comp = struc.composition
        el_dict = comp.get_el_amt_dict()
        arr = []
        for k, v in el_dict.items():
            des = self.get_chgdescrp_arr(k)
            arr.append(des)
        self.chg_stats = descriptor_stats(arr, axis=0).get_stats()

    def get_chgdescrp_arr(self, elm):
        d = ele_data[elm]
        arr = np.ndarray.flatten(np.array(d).astype(float))
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
    charge = Charge(test)
    print(charge.chg_stats)
    print('shape of this descriptor: ', np.shape(charge.chg_stats))

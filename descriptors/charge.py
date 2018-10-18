from pymatgen.core.structure import Structure
import numpy as np
from optparse import OptionParser
import os.path as op
import json

filename = op.join(op.dirname(__file__), 'element_charge.json')


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
        self.mean_chg = np.mean(arr, axis=0)

    def get_chgdescrp_arr(self, elm=''):
        arr = []
        try:
            f = open(filename, 'r')
            emdat = json.load(f)
            f.close()
            arr = emdat[elm][0][1]
        except:
            pass

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
    charge = Charge(test)
    print(charge.mean_chg)
    print('shape of this descriptor: ', np.shape(charge.mean_chg))

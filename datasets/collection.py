from monty.serialization import loadfn
from pymatgen.core.structure import Structure
from optparse import OptionParser

class Collection:
    """Collection of molecular data.
    Used for obtaining pymatgen objects from a small database file.
    """

    def __init__(self, file='sp_metal_aflow_844.json', prop='form_energy_cell'):
        """Create a collection lazily.
        Will read data from json file when needed.
        """

        self.file = file
        self.prop = prop
        self._data = loadfn(file)
        
    def extract_struc_prop(self):
        struc = []
        prop = []
        for dct in self._data:
            try:
                prop.append(dct[self.prop])
                struc.append(Structure(dct['lattice'], dct['atom_array'], dct['coordinates']))
            except:
                pass
        return struc, prop

if __name__ == "__main__":
    # -------------------------------- Options -------------------------
    parser = OptionParser()
    parser.add_option("-f", "--file", dest="file", default='sp_metal_aflow_844.json',
                      help="json file")
    parser.add_option("-p", "--prop", dest="prop", default='form_energy_cell',
                      help="property")

    (options, args) = parser.parse_args()
    struc, prop = Collection(options.file, options.prop).extract_struc_prop()
    print('Reading file from : ', options.file)
    print('The target property is ', options.prop)
    print('Returning {:d} structure'.format(len(struc)))

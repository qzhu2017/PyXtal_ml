from monty.serialization import loadfn
from pymatgen.core.structure import Structure
from optparse import OptionParser

class Collection:
    """Collection of molecular data.
    Used for obtaining pymatgen objects from a small database file.
    """

    def __init__(self, file, prop='formation_energy'):
        """
        1, read the data from json file
        2, extract the structure and property info
        """

        self.file = file
        self.prop = prop
        self._data = loadfn(file)
        
    def extract_struc_prop(self):
        struc = []
        prop = []
        for dct in self._data:
<<<<<<< HEAD
            try:
                prop.append(dct[self.prop])
                struc.append(Structure(dct['lattice'], dct['atom_array'], dct['coordinates']))
            except:
                pass
=======
            # QZ: sometime the property returns None
            if dct[self.prop] is not None:
                struc.append(Structure(dct['lattice'], dct['elements'], dct['coords']))
                prop.append(dct[self.prop])
>>>>>>> 16037eac8747e1fd9739b2fadc379beda46269fa
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


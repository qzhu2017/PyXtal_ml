from RDF import RDF
from ADF import ADF
from chem import Chem
from optparse import OptionParser
import numpy as np
from pymatgen.core.structure import Structure

class descriptor:
    """Collection of molecular data.
    Used for obtaining pymatgen objects from a small database file.

    Example of use:
    >>> from ml_materials.descriptors import descriptor
    >>> des=descriptor(struc, 'all').merge()
    Args:
        name: the type of collection to get. Defaults to "molecules"
    """

    def __init__(self, crystal, libs='all'):
        self.libs = libs
        self.struc = crystal
        self.descriptor = {}
        options = ['RDF', 'ADF', 'Chem']
        self.libs = []
        if libs == 'all':
            self.libs = options
        else:
            for lib in libs.split('+'):
                self.libs.append(lib)

        for lib in self.libs:
            if lib == 'RDF':
                self.descriptor['RDF'] = RDF(self.struc).RDF[1,:]
            elif lib == 'ADF':
                self.descriptor['ADF'] = ADF(self.struc).ADF
            elif lib == 'Chem':
                self.descriptor['Chem'] = Chem(self.struc).mean_chem

    def merge(self):
        arr = []
        for key in self.descriptor.keys():
            if len(arr) == 0:
                arr = self.descriptor[key]
            else:
                arr = np.hstack((arr, self.descriptor[key]))
        return arr

if __name__ == "__main__":
    # -------------------------------- Options -------------------------
    parser = OptionParser()
    parser.add_option("-c", "--crystal", dest="structure", default='',
                      help="crystal from file, cif or poscar, REQUIRED",
                      metavar="crystal")
    parser.add_option("-d", "--descriptors", dest="descriptors", default='all',
                      help="descriptors, all, rdf+adf")

    (options, args) = parser.parse_args()
    if options.structure.find('cif') > 0:
        fileformat = 'cif'
    else:
        fileformat = 'poscar'

    test = Structure.from_file(options.structure)
    des = descriptor(test)
    print(des.libs)
    print(des.merge())
    print('length of the descriptors: ', np.shape(des.merge()))


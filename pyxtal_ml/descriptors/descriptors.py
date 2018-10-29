from pyxtal_ml.descriptors.RDF import RDF
from pyxtal_ml.descriptors.ADF import ADF
from pyxtal_ml.descriptors.chem import Chem
from pyxtal_ml.descriptors.charge import Charge
from pyxtal_ml.descriptors.DDF import DDF
from pyxtal_ml.descriptors.voronoi_descriptors import Voronoi_Descriptors
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
        options = ['RDF', 'ADF', 'DDF', 'Chem', 'Charge', 'Voronoi']
        self.libs = []
        if libs == 'all':
            self.libs = options
        else:
            for lib in libs.split('+'):
                self.libs.append(lib)

        #for lib in self.libs:
        #    if lib in ['Packing_efficiency, Volume_stats, Bond_stats, Coord_number, Chemical_ordering, Environment_attibutes']:
        #        voro = Voronoi_Descriptors(self.struc)
        #        break
        
        for lib in self.libs:
            #voro = Voronoi_Descriptors(self.struc)
            if lib == 'RDF':
                self.descriptor['RDF'] = RDF(self.struc).RDF[1, :]
            elif lib == 'ADF':
                self.descriptor['ADF'] = ADF(self.struc).all
            elif lib == 'DDF':
                self.descriptor['DDF'] = DDF(self.struc).DDF
            elif lib == 'Chem':
                self.descriptor['Chem'] = Chem(self.struc).mean_chem
            elif lib == 'Charge':
                self.descriptor['Charge'] = Charge(self.struc).mean_chg
            elif lib == 'Voronoi':
                self.descriptor['Voronoi'] = Voronoi_Descriptors(self.struc).all()
            #elif lib == 'Packing_efficiency':
            #    self.descriptor['Packing_efficiency'] = voro.get_packing_efficiency(
            #    )
            #elif lib == 'Volume_stats':
            #    self.descriptor['Volume_stats'] = voro.get_volume_statistics()
            #elif lib == 'Bond_stats':
            #    self.descriptor['Bond_stats'] = voro.get_bond_statistics()
            #elif lib == 'Coord_number':
            #    self.descriptor['Coord_number'] = voro.get_effective_coordination_number(
            #    )
            #elif lib == 'Chemical_ordering':
            #    self.descriptor['Chemical_ordering'] = voro.get_chemical_ordering_parameters(
            #    )
            #elif lib == 'Environment_attributes':
            #    self.descriptor['Environment_attributes'] = voro.get_environment_attributes(
            #    )

    def merge(self, keys=None):
        if keys is None:
            keys0 = self.descriptor.keys()
        else:
            keys0 = []
            for key in keys.split('+'):
                keys0.append(key)

        arr = []
        for key in keys0:
            if len(self.descriptor[key]) == 0:
                print(key, np.shape(self.descriptor[key]))
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
    for lib in des.libs:
        print(lib, len(des.descriptor[lib]))
    print(des.merge())
    print('length of the descriptors: ', np.shape(des.merge()))

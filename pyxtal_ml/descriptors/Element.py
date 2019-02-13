from monty.serialization import loadfn
from pymatgen.core.structure import Structure
from pymatgen.core.periodic_table import Element, Specie
import numpy as np
from optparse import OptionParser
import os.path as op

filename = op.join(op.dirname(__file__), 'Elements.json')
ele_data = loadfn(filename)

class element_attributes(object):
    '''
    A class containing selected elemental property information
    relevant to the constituent atoms of a pymatgen structure

    Args:
        crystal: crystal object from pymatgen

    Attributes:
    '''

    def __init__(self, crystal):

        # attributes

        self._crystal = crystal

        # lookup and return elemental attributes
        self.get_elemental_attributes()

    @staticmethod
    def _get_data_from_pymatgen(ele):
        '''
        Get select elemental properties from pymatgen

        args:
            ele: a pymatgen element object

        returns: a 1-d array containing select elemental properties
        '''

        '''
        properties include:
            atomic number, electronegativity, row, group, atomic mass,
            atomic radius, van der waals radius, molar volume,
            thermal conductivity, boiling point, melting point, and
            solid density
        '''
        properties = [ele.Z, ele.X, ele.row, ele.group, ele.atomic_mass,
                      ele.atomic_radius, ele.van_der_waals_radius,
                      ele.molar_volume, ele.thermal_conductivity,
                      ele.boiling_point, ele.melting_point,
                      ele.density_of_solid]

        if None in properties:

            for i, prop in enumerate(properties):

                if prop is None:

                    properties[i] = 0


        # convert to numpy array and return properties
        return np.array(properties)

    @staticmethod
    def _get_data_from_json(ele):
        '''
        Get select elemental properties from json file

        args:
            ele: a pymatgen element object

        returns: a 1-d array containing select elemental properties
        '''

        # convert element object to string
        elm = str(ele)

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


        # call element data from dictionary
        data = ele_data[elm]
        # select property keys
        props = ['first_ion_en', 'elec_aff', 'hfus', 'polzbl']

        # initiate empty array for properties
        properties = []
        for prop in props:
            # call property from dictionary keys
            properties.append(data[prop])

        if None in properties:

            for i, prop in enumerate(properties):

                if prop is None:

                    properties[i] = 0

        # convert to array and return properties
        return np.array(properties)

    def get_elemental_attributes(self):

        # find list of elements through set intersection with itself
        elements = list(set(self._crystal.species).intersection(self._crystal.species))

        # initiate empty list for properties
        arr = []

        for element in elements:
            # append a 1-D property array for each element in the crystal
            composition_fraction = self._crystal.composition.get_atomic_fraction(element)
            arr.append(composition_fraction * np.hstack((self._get_data_from_pymatgen(element),
                                  self._get_data_from_json(element))))

        # convert to numpy array and return
        self.properties = len(elements)*np.array(arr)



if __name__ == "__main__":
    #-------------------------Options------------------
    parser = OptionParser()
    parser.add_option("-c", "--crystal", dest='structure', default='',
                      help='crystal from file, cif or poscar, REQUIRED',
                      metavar='crystal')

    (options, args) = parser.parse_args()

    if options.structure.find('cif') > 0:
        fileformat = 'cif'

    else:
        fileformat = 'poscar'

    test = Structure.from_file(options.structure)

    props = element_attributes(test)

    print(props.properties, np.shape(props.properties), type(props.properties))

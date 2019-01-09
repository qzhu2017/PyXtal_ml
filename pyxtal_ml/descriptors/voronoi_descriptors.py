from pymatgen.analysis.local_env import VoronoiNN
import numpy as np
from itertools import product
from pyxtal_ml.descriptors.angular_momentum import wigner_3j
from pymatgen.core.structure import Structure, IStructure
from pymatgen.core.periodic_table import Element, Specie
from pymatgen.core.bonds import get_bond_length
from pymatgen.analysis.local_env import get_neighbors_of_site_with_index
from scipy.special import sph_harm
from monty.serialization import loadfn
import os.path as op
from optparse import OptionParser

filename = op.join(op.dirname(__file__), 'Elements.json')
ele_data = loadfn(filename)


class Voronoi_Descriptors(object):
    '''
    Computes crystal structure and composition features
    using Voronoi polyhedra

    Args:
        crystal: A pymatgen crystal structure

    Methods:
        get_packing_efficiency:
            calculates the packing efficiency of a cell

        get_volume_statistics:
            calculates the mean absolute deviation of the volume enclosed
            by each voronoi polyhedron in the cell normalized by the mean
            volume

        get_bond_statistics:
            calculates various statistics of the bond lengths and bond length
            variations in cell

        get_effective_coordination_number:
            calculates various statistics about the coordination numbers of all
            constituent elements in the crystal

        get_chemical_ordering_parameters:
            computes the mean chemical ordering parameter and parameter
            variance

        get_environment_attributes:
            Calculates various statistics of selected elemental properties
            associated with each site in the crystal

        q4:
            calculates the Steinhardt bond order parameter q4
            see P. Steinhardt et al. Phys Rev. B 28, 784 198

        q6:
            calculates the Steinhardt bond order parameter q6
            see P. Steinhardt et al. Phys Rev. B 28, 784 19833

        all:
            calculates all of the above and stacks the results into a
            1-d column array


    '''

    def __init__(self, crystal):
        '''
        Assign attributes and compute the Voronoi polyhedra of
        the crystal structure using pymatgen functionality
        '''
        # attributes
        self._crystal = crystal
        self._shells = (1, 2, 3)
        # find elements using set intersection
        self._elements = list(
            set(crystal.species).intersection(crystal.species))

        # call compute polyhedra method to assign polyhedra attribute
        self._compute_polyhedra()

    def all(self):
        '''
        Computes all Voronoi polyhedra features

        Calls all voronoi features and stacks them
        into a 1-d array
        '''
        # call voronoi feature methods
        pef = self.get_packing_efficiency()
        vstat = self.get_volume_statistics()
        ecn = self.get_effective_coordination_number()
        bstat = self.get_bond_statistics()
        cop = self.get_chemical_ordering_parameters()
        ea = self.get_environment_attributes()

        # stack into 1-d array
        arr = np.hstack((pef, vstat, ecn, bstat, cop, ea))

        return arr

    def _populate_element_dict(self):
        '''
        For features that depend on elements, populate a dictionary
        of empty lists with each element in the structure as a key

        Args:
            elements_list: a list of pymatgen element objects

        Returns:
            an dictionary of empty lists where the keys are pymatgen elements

        '''

        # create an empty dictionary
        element_dict = {}
        # iterate over element attribute to create
        # a dictionary of empty lists with the constituent
        # elements as keys
        for element in self._elements:
            element_dict[element] = []

        return element_dict

    @staticmethod
    def _weighted_average(array, weights=None):
        '''
        Compute the weighted average of a 1-d array

        Args:
            array:  a 1-d array or list
            weights: weights corresponding to each element in the list

        Returns:
            the weighted average of the array
        '''

        return np.average(array, weights=weights)

    @staticmethod
    def _MAD(array, weights=None):
        '''
        Compute the mean absolute deviation of a 1-d array

        Args:
            array: A 1-d array or list
            weights:  weights corresponding to each element of the
                      array

        Returns:
            the weighted mean absolute deviation of the 1-d array
        '''

        mean = np.mean(array)
        # mean absolute deviation
        return np.average(np.abs(np.subtract(array, mean)), weights=weights)

    @staticmethod
    def _get_all_nearest_neighbors(method, crystal):
        '''
        Get the nearest neighbor list of a structure

        Args:
            method: method used to compute nearest_neighbors
            crystal:  pymatgen structure

        Returns:
            nearest neighbors
        '''
        # check that the passed crystal is a pymatgen structure object
        if isinstance(crystal, Structure):
            # make the structure an immutable object
            crystal = IStructure.from_sites(crystal)

        # get the voronoi info of the crystal
        return method.get_all_nn_info(crystal)

    @staticmethod
    def _get_chemical_descriptors_from_file(elm, weight):
        '''
        Calls certain elemental properties from the Elements.json file
        to use as weighted chemical environment attributes

        Args:
            elm: a pymatgen element object
            weight: a numerical value to be used as a weight for
                    these attributes

        Returns:
            an array of chemical environment attributes
        '''
        # convert the element object to a string for indexing of the json file
        elm = str(elm)

        # compute the below elemental properties
        properties = ['nsvalence', 'npvalence', 'ndvalence', 'nfvalence',
                      'nsunfill', 'npunfill', 'ndunfill', 'nfunfill',
                      'first_ion_en']

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

        # call the elemental specfic dictionary
        data = ele_data[elm]

        # populate an empty list with the above elemental properties
        arr = []
        for prop in properties:
            # gather the property and multiply by the weight
            arr.append(data[prop] * weight)

        # items 0-3 correspond to the occupancies of the valence structure
        arr += [np.sum(arr[0:4])]  # total valence
        # items 4-7 correspond to the vacancies of the valence structure
        arr += [np.sum(arr[4:8])]  # total unfilled

        return arr

    @staticmethod
    def _get_chemical_descriptors_from_pymatgen(elm, weight):
        '''
        Calls certain elemental properties from Pymatgen to use
        as weighted chemical environment attributes

        Args:
            elm: a pymatgen element object
            weight: a numerical value to be used as a weight
                    for these attributes

        Returns:
            an array of chemical environment attributes
        '''
        element = elm.data
        properties = ['Atomic no', 'Atomic mass', 'row', 'col',
                      'Mendeleev no', 'Atomic radius']

        arr = []

        for prop in properties:
            if prop == 'row':
                arr.append(elm.row * weight)
                continue
            if prop == 'col':
                arr.append(elm.group * weight)
                continue
            else:
                arr.append(element[prop] * weight)

        return arr

    def _get_descr(self, elm, weight):
        '''
        Calls chemical attributes from pymatgen and a json file

        Args:
            elm: a pymatgen element object
            weight: a numerical value to be used as a weight
                    for these attributes

        Returns:
            an array of weighted chemical envronment attributes
        '''

        return (np.hstack((self._get_chemical_descriptors_from_file(elm, weight),
                           self._get_chemical_descriptors_from_pymatgen(elm, weight))))

    def _Get_stats(self, array, weights=None):
        '''
        Compute the min, max, range, mean, and mean absolute deviation
        over a 1-d array

        Args:
            Array: a 1-d float array
            weights: a numerical value to be used as a weight
                     for the chemical environment attributes

        Returns:
            the stats described above
        '''

        return [np.amin(array), np.amax(array), np.ptp(array),
                self._weighted_average(array, weights), self._MAD(array, weights)]

    def _compute_polyhedra(self):
        '''
        Compute the voronoi polyhedron and specie of each atomic site,
        format as a tuple (polyhedron, specie), and store those tuples in
        a list as an attribute
        '''

        # call the voronoi object
        voronoi = VoronoiNN()
        self._polyhedra = []  # declare polyhedra attribute
        # populate the attribute with the polyhedron associated with each
        # atomic site
        for i, _ in enumerate(self._crystal):
            self._polyhedra.append((voronoi.get_voronoi_polyhedra(self._crystal, i),
                                    self._crystal[i].specie))

    def get_packing_efficiency(self):
        '''
        Computes the packing efficiency of the crystal using
        Voronoi polyhedra

        returns:
            [packing efficiency]
        '''
        maximum_radii = []

        # find the minimum distance from the center of the polyhedron
        # to the polyhedron faces
        for polyhedron, element in self._polyhedra:
            radii = []

            for face in polyhedron.values():
                radii.append(face['face_dist'])

            maximum_radii.append(min(radii))
        # sum up all of the sphere volumes corresponding to the
        # minimum face distances
        return [4/3 * np.pi * np.power(maximum_radii, 3).sum()
                / self._crystal.volume]

    def get_volume_statistics(self):
        '''
        Computes the mean volume of all Voronoi polyhedra in the structure

        returns:
            [volume_variance / mean_volume]
        '''
        Volumes = []

        # find the volume associated with each polyhedron
        for polyhedron, element in self._polyhedra:
            volume = []
            for face in polyhedron.values():
                volume.append(face['volume'])

            Volumes.append(np.sum(volume))

        return [self._MAD(Volumes) / np.mean(Volumes)]

    def get_bond_statistics(self):
        '''
        Computes the mean bond length and bond length variance
        of the crystal using Voronoi polyhedra

        returns:
            see Get_stats
        '''

        avg_bond_lengths = []
        bond_length_var = []

        for polyhedron, element in self._polyhedra:
            bond_lengths = []
            face_areas = []
            for face in polyhedron.values():
                # call the bond lengths and face areas (weights)
                bond_lengths.append(face['face_dist'] * 2)
                face_areas.append(face['area'])

            # compute the weighted average of the bond lengths
            mean = self._weighted_average(bond_lengths, face_areas)
            avg_bond_lengths.append(mean)
            # compute the weighted mean absolute deviation of the bond lengths
            # divided by the mean
            bond_length_var.append(self._MAD(bond_lengths, face_areas) / mean)

        # normalize the average bond lengths by the mean
        avg_bond_lengths /= np.mean(avg_bond_lengths)

        ''' compute the mean absolute deviation, max and min of the average
            bond lengths '''
        features = [self._MAD(avg_bond_lengths), np.amin(avg_bond_lengths),
                    np.amax(avg_bond_lengths)]

        features += self._Get_stats(bond_length_var)

        return features

    def get_effective_coordination_number(self):
        '''
        Computes the mean coordination number and coordination number variance
        for all elements in the crystal using Voronoi polyhedra

        returns:
            see Get_stats
        '''
        # populate a dictionary of empty lists with keys
        # for each element in the crystal
        CN_dict = self._populate_element_dict()

        # find the coordination number for each elemet
        for polyhedron, element in self._polyhedra:
            face_area = []
            for face in polyhedron.values():
                face_area.append(face['area'])

            face_area = np.array(face_area)
            # square of the sum divided by the sum of squares
            CN_dict[element].append(
                (face_area.sum()**2)/((face_area**2).sum()))

        CN_eff = []
        ''' populate the a list with the average coordination numbers
            associated with each element '''
        for CN in CN_dict.values():
            CN_eff.append(np.mean(CN))

        return self._Get_stats(CN_eff)

    def get_chemical_ordering_parameters(self):
        '''
        Computes the mean chemical ordering parameter and chemical ordering
        parameter variance for all elements in the crystal using
        Voronoi polyhedra

        returns:
            [mean coordination number, coordination number variance]
        '''
        if len(self._crystal.composition) == 1:
            return [0]*len(self._shells)

        # Get a list of types
        elems, fracs = zip(*self._crystal.composition.element_composition.
                           fractional_composition.items())

        # Precompute the list of NNs in the structure
        weight = 'area'
        voro = VoronoiNN(weight=weight)
        all_nn = self._get_all_nearest_neighbors(voro, self._crystal)

        # Evaluate each shell
        output = []
        for shell in self._shells:
            # Initialize an array to store the ordering parameters
            ordering = np.zeros((len(self._crystal), len(elems)))

            # Get the ordering of each type of each atom
            for site_idx in range(len(self._crystal)):
                nns = voro._get_nn_shell_info(self._crystal, all_nn, site_idx,
                                              shell)

                # Sum up the weights
                total_weight = sum(x['weight'] for x in nns)

                # Get weight by type
                for nn in nns:
                    site_elem = nn['site'].specie.element \
                        if isinstance(nn['site'].specie, Specie) else \
                        nn['site'].specie
                    elem_idx = elems.index(site_elem)
                    ordering[site_idx, elem_idx] += nn['weight']

                # Compute the ordering parameter
                ordering[site_idx, :] = 1 - ordering[site_idx, :] / \
                    total_weight / np.array(fracs)

            # Compute the average ordering for the entire structure
            output.append(np.abs(ordering).mean())

        return output

    def get_environment_attributes(self):
        '''
        Computes some statistics of selected elemental properties
        using voronoi polyhedra

        These statistics include: mean, standard deviation, min, max, and range

        returns:
            a list of property statistics in the crystal
        '''

        attributes = []
        ''' call selected chemical environment attributes
          compute the differences of environment attributes in the structure
          weight those differences using face areas of the polyhedra '''
        for polyhedron, element in self._polyhedra:

            polyhedron_attributes = []
            face_area = []
            element_1 = Element(element)
            for face in polyhedron.values():
                area = face['area']
                element_2 = Element(face['site'].specie)
                face_area.append(area)
                # see get_descr method
                element_1_props = self._get_descr(element_1, area)

                element_2_props = self._get_descr(element_2, area)
                # differences in properties
                polyhedron_attributes.append(
                    (element_1_props - element_2_props))
            # normalize by the surface area of the polyhedron
            surface_area = np.sum(face_area)
            for i, att in enumerate(polyhedron_attributes):
                polyhedron_attributes[i] / surface_area

            # add the attribute to the list of attributes
            attributes += polyhedron_attributes

        # conver to array
        attributes = np.array(attributes)
        delta_stats = []

        # iterate over columns ( each property corresponds to a row )
        for i, _ in enumerate(attributes[0, :]):
            # see get stats
            delta_stats += self._Get_stats(attributes[:, i])

        return delta_stats


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
    voro = Voronoi_Descriptors(test)
    test.make_supercell([2, 2, 2])
    voro = Voronoi_Descriptors(test)
    # print('voro.get_packing_efficiency()')
    # print(voro.get_packing_efficiency())
    # print('voro.get_volume_statistics()')
    # print(voro.get_volume_statistics())
    # print('voro.get_effective_coordination_number()')
    # print(voro.get_effective_coordination_number())
    # print('voro.get_bond_statistics()')
    # print(voro.get_bond_statistics())
    # print('voro.get_chemical_ordering_parameters()')
    # print(voro.get_chemical_ordering_parameters())
    # print('voro.get_environment_attributes()')
    # print(voro.get_environment_attributes())

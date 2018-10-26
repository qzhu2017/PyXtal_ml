from pymatgen.analysis.local_env import VoronoiNN
import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.core.periodic_table import Element
from pymatgen.core.bonds import get_bond_length
from optparse import OptionParser


class Voronoi_Descriptors(object):
    '''
    Computes crystal structure and composition features
    using Voronoi polyhedra

    Args:
        crystal: A pymatgen crystal structure

    Attributes:
        crystal: A pymatgen crystal structure
        comp; composition data of crystal
        elements: the pymatgen element objects of the
                  crystal constituents
        polyhedra: the voronoi polyhedra corresponding to
                   each site in the structure
    '''

    def __init__(self, crystal):
        '''
        Assign attributes and compute the Voronoi polyhedra of
        the crystal structure using pymatgen functionality
        '''
        self.crystal = crystal
        self.comp = crystal.composition
        # find elements using set intersection
        self.elements = list(
            set(crystal.species).intersection(crystal.species))

        self.compute_polyhedra()

    @staticmethod
    def populate_element_dict(elements_list):
        '''
        For features that depend on elements, populate a dictionary
        of empty lists with each element in the structure as a key

        Args:
            elements_list: a list of pymatgen element objects

        Returns:
            an dictionary of empty lists where the keys are pymatgen elements

        '''

        element_dict = {}
        for element in elements_list:
            element_dict[element] = []

        return element_dict

    def compute_polyhedra(self):
        '''
        Compute the voronoi polyhedron and specie of each atomic site,
        format as a tuple (polyhedron, specie), and store those tuples in
        a list as an attribute
        '''

        voronoi = VoronoiNN()
        self.polyhedra = []
        for i, _ in enumerate(self.crystal):
            self.polyhedra.append((voronoi.get_voronoi_polyhedra(self.crystal, i),
                                   self.crystal[i].specie))

    def get_packing_efficiency(self):
        '''
        Computes the packing efficiency of the crystal using
        Voronoi polyhedra

        returns:
            [packing efficiency]
        '''
        maximum_radii = []

        for polyhedron, element in self.polyhedra:
            radii = []

            for face in polyhedron.values():
                radii.append(face['face_dist'])

            maximum_radii.append(min(radii))

        return [4/3 * np.pi * np.power(maximum_radii, 3).sum()
                / self.crystal.volume]

    def get_volume_statistics(self):
        '''
        Computes the mean volume of all Voronoi polyhedra in the structure

        returns:
            [mean_volume, volume_variance]
        '''
        Volume = []

        for polyhedron, element in self.polyhedra:
            volume = []
            for face in polyhedron.values():
                volume.append(face['volume'])

            Volume.append(np.sum(volume))

        return [np.mean(Volume), np.var(Volume)]

    def get_bond_statistics(self):
        '''
        Computes the mean bond length and bond length variance
        of the crystal using Voronoi polyhedra

        returns:
            [mean bond length, bond lenth variance]
        '''

        bond_lengths = []

        for polyhedron, element in self.polyhedra:
            bond_lengths = []
            for face in polyhedron.values():
                bond_lengths.append(get_bond_length(element,
                                                    face['site'].specie))

        mean_bond_lengths = np.mean(bond_lengths)
        bond_length_variance = np.var(bond_lengths)

        return [mean_bond_lengths, bond_length_variance]

    def get_effective_coordination_number(self):
        '''
        Computes the mean coordination number and coordination number variance
        for all elements in the crystal using Voronoi polyhedra

        returns:
            [mean coordination number, coordination number variance]
        '''
        CN_dict = self.populate_element_dict(self.elements)

        for polyhedron, element in self.polyhedra:
            face_area = []
            for face in polyhedron.values():
                face_area.append(face['area'])

            face_area = np.array(face_area)
            CN_dict[element].append(
                (face_area.sum()**2)/((face_area**2).sum()))

        CN_eff = []

        for CN in CN_dict.values():
            CN_eff.append(np.mean(CN))

        return [np.mean(CN_eff), np.var(CN_eff)]

    def get_chemical_ordering_parameters(self):
        '''
        Computes the mean chemical ordering parameter and chemical ordering
        parameter variance for all elements in the crystal using
        Voronoi polyhedra

        returns:
            [mean coordination number, coordination number variance]
        '''
        alphas = []
        Chemical_Ordering = self.populate_element_dict(self.elements)
        for polyhedron, element in self.polyhedra:
            face_area = []
            atomic_fraction = []
            ordering_faces = []
            for face in polyhedron.values():
                face_area.append(face['area'])
                if element == face['site'].specie:
                    atomic_fraction.append(
                        self.comp.get_atomic_fraction(element))
                    ordering_faces.append(face['area'])

            ordering_faces = np.array(ordering_faces)
            atomic_fraction = np.array(atomic_fraction)
            face_area = np.array(face_area)
            Chemical_Ordering[element].append(np.mean(1-ordering_faces /
                                                      atomic_fraction /
                                                      face_area.sum()))

            for alpha in Chemical_Ordering.values():
                if len(alpha) == 0:
                    alphas.append(1)
                else:
                    alphas.append(np.mean(alpha))

            return [np.mean(alphas), np.var(alphas)]

    def get_environment_attributes(self):
        '''
        Computes some statistics of selected elemental properties
        using voronoi polyhedra

        These statistics include: mean, standard deviation, min, max, and range

        returns:
            a list of property statistics in the crystal
        '''

        delta_dict = {'an': [], 'aw': [], 'rn': [], 'cn': [],
                      'mn': [], 'ar': []}
        # implement valence

        for polyhedron, element in self.polyhedra:
            atomic_number = []
            atomic_weight = []
            row_number = []
            column_number = []
            mendeleev_number = []
            atomic_radii = []

            face_area = []
            element_1 = Element(element)
            for face in polyhedron.values():
                area = face['area']
                element_2 = Element(face['site'].specie)
                face_area.append(area)
                atomic_number.append(area*(element_2.data['Atomic no'] -
                                           element_1.data['Atomic no']))
                atomic_weight.append(area*(element_2.data['Atomic mass'] -
                                           element_1.data['Atomic mass']))
                row_number.append(area*(element_2.row - element_1.row))
                column_number.append(area*(element_2.group -
                                           element_1.group))
                mendeleev_number.append(area*(element_2.data['Mendeleev no'] -
                                              element_1.data['Mendeleev no']))
                atomic_radii.append(area*(element_2.data['Atomic radius'] -
                                          element_1.data['Atomic radius']))

            surface_area = np.sum(face_area)
            delta_dict['an'].append(np.sum(atomic_number) / surface_area)
            delta_dict['aw'].append(np.sum(atomic_weight) / surface_area)
            delta_dict['rn'].append(np.sum(row_number) / surface_area)
            delta_dict['cn'].append(np.sum(column_number) / surface_area)
            delta_dict['mn'].append(np.sum(mendeleev_number) / surface_area)
            delta_dict['ar'].append(np.sum(atomic_radii) / surface_area)

        delta_stats = []
        for delta in delta_dict.values():
            delta_stats.append(np.mean(delta))
            delta_stats.append(np.std(delta))
            dmax, dmin = np.amax(delta), np.amin(delta)
            delta_stats.append(dmax)
            delta_stats.append(dmin)
            delta_stats.append(dmax-dmin)

        return delta_stats

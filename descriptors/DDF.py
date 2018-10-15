import numpy as np
from scipy.spatial.distance import cdist
import pymatgen.core.periodic_table as per
from pymatgen.core.composition import Composition
import itertools


class DDF(object):
    '''
    A class representing the Covalent and Metallic
    Bond Angle Distribution functions

    Attributes:
        self:  The dihedral distribution function
        crystal:  a pymatgen structure
        R_max: the maximum distance in angstroms for the find all neighbors function
        Span:  decision factor for determining if a bond exists.

    '''

    def __init__(self, crystal, R_max=12, Span=0.18):
        '''DDF Object

        Computes the dihedral distribution function '''
        self.R_max = R_max
        self.Span = Span

        self.compute_DDF(crystal)

    def compute_DDF(self, crystal):
        '''
        Computes the dihedral distribution function from a pymatgen struc

        Args:
            crystal: a pymatgen structure

        Returns:
            the dihedral distribution function
        '''

        R_max = self.R_max # maximal radius

        Span = self.Span  # bond determinant, will change later

        # see structure array method
        struc_array = self.structure_array(crystal, R_max)

        # see compute_bond_angles method
        bond_angles = self.compute_bond_angles(struc_array, Span)
        print(bond_angles)
        #  compute the histogram of the DDF
        bins = np.arange(0 ,181,1)
        nbins = np.arange(1,181,1)
        ang_hist, ang_bins = np.histogram(bond_angles,bins=bins)
        self.DDF = np.vstack((nbins, ang_hist))

    def create_label_list(self, crystal):
        '''
        Populates a dictionary of the crystal constituents

        Args:
            crystal:  A pymatgen structure

        Returns:
            label_list: a dictonary of 3 empty lists indexed by constituent elements'''


        elements = list(set(crystal.species).intersection(crystal.species))

        constituents = [str(element) for element in elements]

        label_list = {}

        for element in constituents:
            label_list[element] = [[], [], []]

        return label_list

    def structure_array(self, crystal, R_max):
        '''
        Populates the label list with cartesian coordinates and bond radii,
        then organizes that information into an array for computation

        Args:
            crystal:  A pymatgen structure
            R_max: the maximum distances for the get all neighbors function

        Returns:
            an array where the first three colums are the x, y, and z coordinates
            and the last two columns correspond to the atomic and metallic radii
                '''
        elements_dict = self.create_label_list(crystal)

        neighbor = crystal.get_all_neighbors(r=R_max, include_index=True, include_image=True)

        for i, _ in enumerate(neighbor):
            for j, _ in enumerate(neighbor[i]):
                element = neighbor[i][j][0].species_string
                ele = per.Element(element)
                elements_dict[element][0].append(neighbor[i][j][0].coords)
                elements_dict[element][1].append(ele.atomic_radius)
                if (ele.is_post_transition_metal or ele.is_transition_metal) is True:
                    elements_dict[element][2].append(ele.metallic_radius)
                else:
                    elements_dict[element][2].append(np.nan)

        for i, element in enumerate(elements_dict):
            if i == 0:
                coord_array = np.array(elements_dict[element][0])
                covalent_radius_array = np.array(elements_dict[element][1])[:, np.newaxis]
                metallic_radius_array = np.array(elements_dict[element][2])[:, np.newaxis]
            else:
                coord_array = np.vstack((coord_array, np.array(elements_dict[element][0])))
                covalent_radius_array = np.vstack((covalent_radius_array, np.array(elements_dict[element][1])[:, np.newaxis]))
                metallic_radius_array = np.vstack((metallic_radius_array, np.array(elements_dict[element][2])[:, np.newaxis]))

        return np.hstack((coord_array, covalent_radius_array, metallic_radius_array))

    def compute_bond_angles(self, struc_array, Span):
        '''
        Computes the dihedral bond angles in a crystal

        Args:
            Struc_array:  see structure_array method
            span:  see init

        Returns:  a list of dihedral angles for covalent and metallic bonds.'''

        distance_array = cdist(struc_array[:, 0:3], struc_array[:, 0:3])

        bond_angles = []

        for i, coord0 in enumerate(struc_array):
            covalent_bond_coords = []
            metallic_bond_coords = []
            covalent_bond_sum = 0
            metallic_bond_sum = 0

            for j, coord1 in enumerate(struc_array):
                if (coord0[3] + coord1[3] - Span) < distance_array[i][j] < (coord0[3] + coord1[3] + Span):
                    covalent_bond_coords.append(coord1[0:3]-coord0[0:3])

                if (coord0[4] + coord1[4] - Span) < distance_array[i][j] < (coord0[4] + coord1[4] + Span):
                    metallic_bond_coords.append(coord1[0:3]-coord0[0:3])

            covalent_bond_sum = len(covalent_bond_coords)
            metallic_bond_sum = len(metallic_bond_coords)

            if covalent_bond_sum > 2:
                for bonds in itertools.combinations(covalent_bond_coords, 3):
                    plane_1 = np.cross(bonds[0], bonds[1])
                    plane_2 = np.cross(bonds[0], bonds[2])
                    covalent_angle = np.degrees(
                                                np.arccos((np.dot(plane_1, plane_2)/
                                                           np.linalg.norm(plane_1)/
                                                           np.linalg.norm(plane_2))))

                    if np.isnan(covalent_angle) == False and covalent_angle > 1e-2:
                        bond_angles.append(covalent_angle)

            if metallic_bond_sum > 2:
                for bonds in itertools.combinations(metallic_bond_coords, 3):

                    plane_1 = np.cross(bonds[0], bonds[1])
                    plane_2 = np.cross(bonds[0], bonds[2])

                    metallic_angle = np.degrees(
                                                np.arccos((np.dot(plane_1, plane_2)/
                                                           np.linalg.norm(plane_1)/
                                                           np.linalg.norm(plane_2))))

                    if np.isnan(metallic_angle) == False and metallic_angle > 1e-2:
                        bond_angles.append(metallic_angle)

        return bond_angles

import numpy as np
from scipy.spatial.distance import cdist
import pymatgen.core.periodic_table as per
import itertools
import matplotlib.pyplot as plt


class DDF(object):
    '''
    A class representing the Covalent and Metallic
    Dihedral bond Angle Distribution functions

    Args:
        crystal:  a pymatgen structure
        R_max: the maximum distance in angstroms for the find all neighbors
               function
        Span:  decision factor for determining if a bond exists.
        bin_width: the width of the bins in the distribution
    Attributes:
        label_list: the label list containing the cartesian coordinates,
                    atomic/metallic radii of each element
                    struc_array: the looping array for angle computation
        angles: dihedral angles
        bins:  angle bins
        DDF: the dihedral angle distribution
    '''

    def __init__(self, crystal, R_max=4, Span=0.18, bin_width=1.0):
        '''DDF Object

        Computes the dihedral distribution function '''
        self.R_max = R_max
        self.Span = Span
        self.Width = bin_width

        self.compute_DDF(crystal)

    def compute_DDF(self, crystal):
        '''
        Computes the dihedral distribution function from a pymatgen struc

        Args:
            crystal: a pymatgen structure

        Returns:
            the dihedral distribution function
        '''

        # see compute_bond_angles method
        self.compute_bond_angles(crystal)

        bins = np.arange(self.Width, 181, self.Width)

        self.DDF, self.bins = np.histogram(self.angles, bins=bins,
                                           density=True)

    def create_label_list(self, crystal):
        '''
        Populates a dictionary of the crystal constituents

        Args:
            crystal:  A pymatgen structure

        Returns:
            label_list: a dictonary of 3 empty lists indexed by constituent
                        elements'''

        # Use a set intersection from the specii to determine the elements
        # in the crystal
        elements = list(set(crystal.species).intersection(crystal.species))

        # convert the specie set to a string list
        constituents = [str(element) for element in elements]

        # create an empty dictionary
        label_list = {}

        # populate the dictionary with element labels
        for element in constituents:
            label_list[element] = [[], [], []]
        self.label_list = label_list

    def structure_array(self, crystal):
        '''
        Populates the label list with cartesian coordinates and bond radii,
        then organizes that information into an array for computation

        Args:
            crystal:  A pymatgen structure
            R_max: the maximum distances for the get all neighbors function

        Returns:
            an array where the first three colums are the x, y, and z
            coordinates and the last two columns correspond to the
            atomic and metallic radii
                '''

        # create label list
        self.create_label_list(crystal)

        elements_dict = self.label_list

        # get all neighbors out to R_max
        neighbor = crystal.get_all_neighbors(r=self.R_max, include_index=True,
                                             include_image=True)

        # loop over all neighbors to each atom
        for i, _ in enumerate(neighbor):  # origin atom
            for j, _ in enumerate(neighbor[i]):  # neighbors
                # call string to index dictionary
                element = neighbor[i][j][0].species_string
                # call element object for atomic and metallic radii
                ele = per.Element(element)
                # input the cartesian coords, atomic radii, and metallic
                # radii information into the label list
                elements_dict[element][0].append(neighbor[i][j][0].coords)
                elements_dict[element][1].append(ele.atomic_radius)

                # check if the element is a transition metal
                if (ele.is_post_transition_metal or
                        ele.is_transition_metal) is True:

                    elements_dict[element][2].append(ele.metallic_radius)
                else:
                    elements_dict[element][2].append(np.nan)

        self.label_list = elements_dict

        #  structure label list into array
        for i, element in enumerate(elements_dict):
            # first loop populate the array with the first element
            if i == 0:
                coord_array = np.array(elements_dict[element][0])
                covalent_radius_array = np.array(elements_dict[element][1]
                                                 )[:, np.newaxis]
                metallic_radius_array = np.array(elements_dict[element][2]
                                                 )[:, np.newaxis]

            else:  # stack the array with succeeding elements
                coord_array = np.vstack((coord_array,
                                         np.array(elements_dict[element][0])))
                covalent_radius_array = np.vstack((covalent_radius_array,
                                                   np.array(
                                                       elements_dict[
                                                           element][1]
                                                   )[:, np.newaxis]))
                metallic_radius_array = np.vstack((metallic_radius_array,
                                                   np.array(
                                                       elements_dict[
                                                           element][2]
                                                   )[:, np.newaxis]))

        self.struc_array = np.hstack((coord_array, covalent_radius_array,
                                      metallic_radius_array))

    def compute_bond_angles(self, crystal):
        '''
        Computes the dihedral bond angles in a crystal

        Args:
            Struc_array:  see structure_array method
            span:  see init

        Returns:  a list of dihedral angles for covalent and metallic bonds.'''

        Span = self.Span

        self.structure_array(crystal)
        struc_array = self.struc_array
        # compute the distances between each atomic pair
        distance_array = cdist(struc_array[:, 0:3], struc_array[:, 0:3])

        # declare an empty list to store bond angles
        bond_angles = []

        for i, coord0 in enumerate(struc_array):  # origin atom
            # allocate initial values for position vectors and bond sums
            covalent_bond_vectors = []
            metallic_bond_vectors = []
            covalent_bond_sum = 0
            metallic_bond_sum = 0

            for j, coord1 in enumerate(struc_array):  # neighbors
                # check if a covalent bond exists
                if (coord0[3] + coord1[3] - Span) < distance_array[i][j] < (
                       coord0[3] + coord1[3] + Span):

                    # append covalent bond position vectors
                    covalent_bond_vectors.append(coord1[0:3]-coord0[0:3])

                # check if a metallic bond exists
                if (coord0[4] + coord1[4] - Span) < distance_array[i][j] < (
                        coord0[4] + coord1[4] + Span):

                    # append metallic bond position vectors
                    metallic_bond_vectors.append(coord1[0:3]-coord0[0:3])

            # sum the number of bonds
            covalent_bond_sum = len(covalent_bond_vectors)
            metallic_bond_sum = len(metallic_bond_vectors)

            # can only create dihedral planes if there are 3 bonds or more
            if covalent_bond_sum > 2:
                # iterate over position vector triples to find dihedral angles
                for bonds in itertools.combinations(covalent_bond_vectors, 3):
                    # create two places using cross product of
                    # position vectors
                    plane_1 = np.cross(bonds[0], bonds[1])
                    plane_2 = np.cross(bonds[0], bonds[2])

                    # find the angle between the two planes
                    covalent_angle = np.degrees(
                                                np.arccos((np.dot(plane_1,
                                                                  plane_2) /
                                                           np.linalg.norm(
                                                               plane_1) /
                                                           np.linalg.norm(
                                                               plane_2))))

                    # only append bond if it is certainly not zero
                    if (np.isnan(covalent_angle) == False and
                            covalent_angle > 1e-2):
                        bond_angles.append(covalent_angle)

            # can only create dihedral planes if there are 3 bonds or more
            if metallic_bond_sum > 2:
                # iterate over position vector triples to find dihedral angles
                for bonds in itertools.combinations(metallic_bond_vectors, 3):
                    # create two places using cross product of
                    # position vectors
                    plane_1 = np.cross(bonds[0], bonds[1])
                    plane_2 = np.cross(bonds[0], bonds[2])

                    # find the angle between the two planes
                    metallic_angle = np.degrees(
                                                np.arccos((np.dot(plane_1,
                                                                  plane_2) /
                                                           np.linalg.norm(
                                                               plane_1) /
                                                           np.linalg.norm(
                                                               plane_2))))

                    # only append bond if it is certainly not zero
                    if (np.isnan(metallic_angle) == False and
                            metallic_angle > 1e-2):
                        bond_angles.append(metallic_angle)

        self.angles = bond_angles

    def plot_DDF(self, filename=None):
        '''
        Plots the dihedral angle distribution function

        Args:
            filename: the path to safe the file'''
        plt.hist(self.angles, self.bins,
                 density=True)
        plt.grid()
        plt.xlabel("Angle (degree)", fontsize=16)
        plt.ylabel("DDF", fontsize=16)
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename)
            plt.close()

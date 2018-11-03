from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.periodic_table import Element, Specie
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d
from monty.serialization import loadfn
import itertools
import os.path as op
from optparse import OptionParser

filename = op.join(op.dirname(__file__), 'element_charge.json')
ele_data = loadfn(filename)


class PRDF(object):
    '''
    Computes the pairwise RDF of a given crystal structure

    Args:
        crystal: A pymatgen crystal structure
        symmetrize: bool, whether or not to symetrize the structure
                    before computation
        R_max: the cutoff distance
        R_bin: bin length when computing the RDF
        sigma: width of gaussian smearing

    Attributes:
        prdf: the pairwise radial distribution function
    '''

    def __init__(self, crystal, symmetrize=True,
                 R_max=6, R_bin=0.2, sigma=0.2):
        '''
        '''
        # populate the private attributes
        self._R_max = R_max
        self._R_bin = R_bin
        self._sigma = sigma
        self._elements = list(
            set(crystal.species).intersection(crystal.species))

        if symmetrize:
            finder = SpacegroupAnalyzer(crystal, symprec=0.06,
                                        angle_tolerance=5)
            crystal = finder.get_conventional_standard_structure()

        self._crystal = crystal

        self.create_RDF_table()

        self.compute_PRDF()

    def create_RDF_table(self):
        '''
        Creates a dictionary with pairwise element combination keys
        with R_max / R_bin length zero vectors.
        '''
        self.prdf_dict = {}
        elements = []
        for element in ele_data.keys():
            elements.append(str(element))

        # all possible pairwise combinations without repeated entries
        combs = itertools.combinations_with_replacement(elements, 2)

        for comb in combs:
            if comb[0] < comb[1]:
                self.prdf_dict[comb[0]+'-'+comb[1]
                               ] = np.zeros(round(self._R_max / self._R_bin))

            else:
                self.prdf_dict[comb[1]+'-'+comb[0]
                               ] = np.zeros(round(self._R_max / self._R_bin))

    def compute_PRDF(self):
        '''

        '''

        neighbors = self._crystal.get_all_neighbors(self._R_max)

        elements = [str(ele) for ele in self._elements]

        distances = {}

        combs = itertools.combinations_with_replacement(elements, 2)

        for comb in combs:
            if comb[0] <= comb[1]:
                distances[comb[0]+'-'+comb[1]] = []

            else:
                distances[comb[1]+'-'+comb[0]] = []

        for i, site in enumerate(self._crystal):
            ele_1 = self._crystal[i].species_string
            for j, neighbor in enumerate(neighbors):
                ele_2 = neighbors[i][j][0].species_string

                if ele_1 <= ele_2:
                    comb = ele_1+'-'+ele_2
                    distances[comb].append(neighbors[i][j][1])

                else:
                    comb = ele_2+'-'+ele_1
                    distances[comb].append(neighbors[i][j][1])

        bins = np.arange(0, self._R_max+self._R_bin, self._R_bin)

        shell_volume = 4/3 * np.pi * (np.power(bins[1:], 3) -
                                      np.power(bins[:-1], 3))

        site_density = self._crystal.num_sites / self._crystal.volume

        neighbors_length = len(neighbors)

        for comb in distances.keys():

            hist, _ = np.histogram(distances[comb], bins, density=True)

            self.prdf_dict[comb] = (hist / shell_volume / site_density /
                                    neighbors_length)

        for i, PRDF in enumerate(self.prdf_dict.items()):
            if i == 0:
                self.PRDF = PRDF[1]
            else:
                self.PRDF = np.hstack((self.PRDF, PRDF[1]))

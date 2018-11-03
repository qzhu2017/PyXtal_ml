from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.periodic_table import Element, Specie
import numpy as np
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

    Attributes:
        PRDF: the pairwise radial distribution function
        prdf_dict: the dictionary of pairwise radial distribution
                   functions
    '''

    def __init__(self, crystal, symmetrize=True,
                 R_max=6, R_bin=0.2):
        '''
        '''
        # populate the private attributes
        self._R_max = R_max
        self._R_bin = R_bin

        # create a list of constituent element objects
        self._elements = list(
            set(crystal.species).intersection(crystal.species))

        # symmetrize the structure
        if symmetrize:
            finder = SpacegroupAnalyzer(crystal, symprec=0.06,
                                        angle_tolerance=5)
            crystal = finder.get_conventional_standard_structure()

        # populate private crystal attribute
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

        '''populate every possible element pairwise combination with each
           combination in alphabetical order '''
        for comb in combs:

            '''the if and else condtions ensure that the elements are in
               alphabetical order

               the pairwise element combinations ( Bi-Te ) are used as keys to
               access a dictionary of pairwise RDF's corresponding to each
               every possible elemental pair, all RDF's are populated
               initially as zero vectors of length R_max / R_bin'''
            if comb[0] <= comb[1]:
                self.prdf_dict[comb[0]+'-'+comb[1]
                               ] = np.zeros(round(self._R_max / self._R_bin))

            else:
                self.prdf_dict[comb[1]+'-'+comb[0]
                               ] = np.zeros(round(self._R_max / self._R_bin))

    def compute_PRDF(self):
        '''
        Compute the pairwise radial distribution functions of all possible
        combinations of constituent elements in the given crystal structure
        '''

        # get all neighbors up to R_max
        neighbors = self._crystal.get_all_neighbors(self._R_max)

        '''convert elements list from element objects to strings for indexing
           the prdf dictionary'''
        elements = [str(ele) for ele in self._elements]

        '''Populate a dictionary of empty lists with keys given by all
           possible element combinations in the crystal structure
           This dictionary will be used to store pairwise distances'''
        distances = {}
        combs = itertools.combinations_with_replacement(elements, 2)
        for comb in combs:
            '''conditions ensure that dictionary keys are ordered
               alphabetically according to the element symbols'''
            if comb[0] <= comb[1]:
                distances[comb[0]+'-'+comb[1]] = []
            else:
                distances[comb[1]+'-'+comb[0]] = []

        '''populate the pairwise distance dictionary using the element
           species at the origin site and all neighbor sites

           the distances are called from the 2nd element of the tuple
           neighbors[i][j][1]
           the first element in the tuple is the site information'''
        for i, site in enumerate(self._crystal):
            ele_1 = self._crystal[i].species_string
            for j, neighbor in enumerate(neighbors):
                ele_2 = neighbors[i][j][0].species_string
                '''again the conditions ensure that the element combinations
                   are ordered alphabetically'''
                if ele_1 <= ele_2:
                    comb = ele_1+'-'+ele_2
                    distances[comb].append(neighbors[i][j][1])
                else:
                    comb = ele_2+'-'+ele_1
                    distances[comb].append(neighbors[i][j][1])

        # distance bins used for the pairwise RDF
        bins = np.arange(0, self._R_max+self._R_bin, self._R_bin)

        '''compute the shell volume using the first and last element
           of the distance bins'''
        shell_volume = 4/3 * np.pi * (np.power(bins[1:], 3) -
                                      np.power(bins[:-1], 3))

        # compute the site density using pymatgen structure attributes
        site_density = self._crystal.num_sites / self._crystal.volume

        # length of neighbors array (the number of atoms in the primitive cell)
        neighbors_length = len(neighbors)

        '''populate the prdf_dict with the pairwise rdfs associated with the
           distance information in the distance dictionary'''
        for comb in distances.keys():
            '''use numpy's histogram function to find a propability density
               associated with each bin for the RDF'''
            hist, _ = np.histogram(distances[comb], bins, density=True)

            # RDF = counts / (volume * site density * sites in primitive cell)
            self.prdf_dict[comb] = (hist / shell_volume / site_density /
                                    neighbors_length)

        '''stack all prdf arrays so that the descriptor length
           is invariant between crystal structure so long as
           r_max and r_bin are held constant'''
        for i, PRDF in enumerate(self.prdf_dict.values()):
            if i == 0:
                self.PRDF = PRDF
            else:
                self.PRDF = np.hstack((self.PRDF, PRDF))


if __name__ == "__main_":
    # ---------------------- Options ------------------------
    parser = OptionParser()
    parser.add_option("-c", "--crystal", dest="structure", default='',
                      help="crystal from file, cif or poscar, REQUIRED",
                      metavar="crystal")
    parser.add_option("-r", "--Rmax", dest="Rmax", default=6, type='float',
                      help="Rmax, default: 6 A", metavar="Rmax")
    parser.add_option("-d", "--delta", dest="delta", default=0.2,
                      type='float', help="step length, default: 0.2",
                      metavar="R_bin")

    (options, args) = parser.parse_args()
    if options.structure.find('cif') > 0:
        fileformat = 'cif'
    else:
        fileformat = 'poscar'

    test = Structure.from_file(options.structure)
    prdf = PRDF(test, R_max=options.Rmax, R_bin=options.delta)
    print(prdf.PRDF)
    test.make_supercell([2, 2, 2])
    prdf = PRDF(test)
    print(prdf.PRDF)

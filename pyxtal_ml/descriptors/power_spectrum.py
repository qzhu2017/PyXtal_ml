import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.analysis.local_env import get_neighbors_of_site_with_index
import pyxtal_ml.descriptors.bond_order_params as bop
from optparse import OptionParser


class power_spectrum(object):
    '''
    Computes the power spectrum corresponding to each periodic site
    in a pymatgen structure using the steinhardt bond order
    parameters

    Args:
        crystal: A pymatgen crystal structure
        L: maximum degree of order parameter
    '''

    def __init__(self, crystal, L=6):
        # all power spectrum values up to maximum degree L
        ls = np.arange(2, L+1, 1)
        power_spectrum = self._populate_dicts(ls)
        '''Loop over l to calculate each power spectrum parameter'''
        for l in ls:
            parameter_index = str(l)
            for index, site in enumerate(crystal):
                # get all nearest neighbors
                neighbors = get_neighbors_of_site_with_index(
                    crystal, index, approach='voronoi')
                # generate free integer paramters
                # calculate power spectrum pl see Pl method
                power_spectrum['p' +
                               parameter_index].append(self.Pl(site, neighbors, l))

        spectrum_values = list(power_spectrum.values())

        '''
        If the list is 1 dimensional simply take the mean of the list
        If the list is 2 dimensional take the mean over the rows of the list'''
        self.Power_spectrum = np.array(spectrum_values).T

    def Pl(self, site, neighbors, l):
        '''
        Computes the power spectrum of a neighbor environment
        using Steinhardt Bond Order Parameters

        Args:
            site: a pymatgen periodic site
            neighbors: a list of pymatgen periodic sites
                       corresponding to the origin sites
                       nearest neighbors
            l:  free integer parameter

        Returns:
            pl: the power spectrum value pl of a periodic site, float
        '''
        # the closed set of integers [-l,l]
        mvals = self._mvalues(l)
        # complex vector of all qlm values
        qlms = bop._qlm(site, neighbors, l, mvals)
        # scalar product of complex vector
        dot = bop._scalar_product(qlms, qlms)
        # steinhardt bond order parameter ql
        ql = bop._ql(dot, l)
        # compute Power spectrum element
        Pl = (2*l + 1) / (4 * np.pi) * ql**2
        return Pl

    @staticmethod
    def _mvalues(l):
        '''Returns the closed set of integers [-l,l]'''
        return np.arange(-l, l+1, 1)

    def _populate_dicts(self, ls):
        '''
        Populates a dictionary for all steinhardt order parameters
        q0, w0, q1, w1 , ... , ql, ql
        '''
        parameter_dict = {}
        for l in ls:
            index = str(l)
            parameter_dict['p' + index] = []
        return parameter_dict


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
    x = power_spectrum(test).Power_spectrum
    print(x, np.shape(x))

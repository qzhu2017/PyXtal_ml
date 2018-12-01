from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.periodic_table import Element, Specie
from pymatgen.analysis.local_env import get_neighbors_of_site_with_index
import numpy as np
import itertools
from sympy.physics.quantum.cg import CG
from sympy.physics.quantum.spin import Rotation
from sympy import S
from optparse import OptionParser


class Bispectrum(object):

    def __init__(self, crystal, j_max=5, cutoff_radius=6.5, symmetrize=True):
        '''
        '''

        # populate private attributes
        self._j_max = j_max
        self._Rc = cutoff_radius

        self._factorial = [1]

        for i in range(int(3. * j_max) + 2):
            if i > 0:
                self._factorial += [i * self._factorial[i-1]]

        # symmetrize structure option
        if symmetrize:
            finder = SpacegroupAnalyzer(crystal, symprec=0.06,
                                        angle_tolerance=5)
            crystal = finder.get_conventional_standard_structure()

        self._crystal = crystal

        '''
        Here we iterate over each site and its corresponding neighbor
        environment to compute the bispectrum coefficients of each
        site-neighbor interaction.

        At each site, the bispectrum is a list where the list elements are
        the bispectrum corresponds each to a site-neighbor interaction

        In turn the descriptor is effectively a list of lists, where the
        element lists contain the bispectrum coefficients of the neighbor
        environment
        '''
        self.bispectrum = []
        for index, site in enumerate(crystal):
            site_neighbors = get_neighbors_of_site_with_index(
                crystal, index, approach='voronoi')
            # site loops over central atoms
            site_bispectrum = []
            for _2j1 in range(2 * self._j_max + 1):
                j1 = _2j1/2
                j2 = _2j1/2
                for j in range(int(min(_2j1, self._j_max,)) + 1):
                    term = self.calculate_B(
                        j1, j2, 1.0*j, site, site_neighbors)
                    term = term.real
                    site_bispectrum += [term]
            self.bispectrum += [site_bispectrum]

    @staticmethod
    def _m_values(k):
        assert k >= 0

        return [k - i for i in range(int(2 * k + 1))]

    def calculate_B(self, j1, j2, j, site, site_neighbors):
        '''
        Calculates the bispectrum coefficients associated with
        atomic site in a crystal structure

        Args:
            j1: index for first summnation
            j2: index for second summnation
            j: index of spherical harmonic expansion
            site: pymatgen site for central atom
            site_neighbors: pymatgen site neighbor list
                corresponding to the central atom

        Returns:
            Bispectrum coefficients: a list of floats
        '''
        B = 0.
        mvals = self._m_values(j)
        for m in mvals:
            print(m)
            for m_prime in mvals:
                c = self.calculate_c(j, m_prime, m, site, site_neighbors)
                m1bound = min(j1, m + j2)
                m_prime1_bound = min(j1, m_prime + j2)
                m1 = max(-j1, m-j2)
                while m1 < (m1bound + 0.5):
                    m_prime1 = max(-j1, m_prime - j2)
                    while m_prime1 < (m_prime1_bound + 0.5):
                        c1 = self.calculate_c(j1, m_prime1, m1, site,
                                              site_neighbors)
                        c2 = self.calculate_c(j2, m_prime-m_prime1, m-m1, site,
                                              site_neighbors)

                        B += float(CG(j1, m1, j2, m-m1, j, m).doit()) * \
                            float(CG(j1, m_prime1, j2, m_prime-m_prime1, j, m_prime).doit()) * \
                            np.conjugate(c) * c1 * c2
                        m_prime1 += 1
                    m1 += 1

        return B

    def calculate_c(self, j, m_prime, m, site, site_neighbors):
        value = 0.
        for neighbor in site_neighbors:
            x = neighbor.coords[0] - site.coords[0]
            y = neighbor.coords[1] - site.coords[1]
            z = neighbor.coords[2] - site.coords[2]
            r = np.linalg.norm(neighbor.coords - site.coords)
            if r > 10.**(-10.):

                psi = np.arcsin(r / self._Rc)

                theta = np.arccos(z / r)
                if abs((z / r) - 1.0) < 10.**(-8.):
                    theta = 0.0
                elif abs((z / r) + 1.0) < 10.**(-8.):
                    theta = np.pi

                if x < 0.:
                    phi = np.pi + np.arctan(y / x)
                elif 0. < x and y < 0.:
                    phi = 2 * np.pi + np.arctan(y / x)
                elif 0. < x and 0. <= y:
                    phi = np.arctan(y / x)
                elif x == 0. and 0. < y:
                    phi = 0.5 * np.pi
                elif x == 0. and y < 0.:
                    phi = 1.5 * np.pi
                else:
                    phi = 0.

                # populate this with atomic numbers
                value += 1.0 * \
                    np.conjugate(self.U(j, m, m_prime, psi, theta, phi)) * \
                    self.cutoff_function(r, self._Rc)

            else:
                continue

        return value

    def U(self, j, m, m_prime, psi, theta, phi):
        result = 0.
        mvals = self._m_values(j)
        for mp in mvals:
            result += complex(Rotation.D(S(int(2*j))/2, S(int(2*m))/2, S(int(2*mp))/2, phi, theta, -phi).doit()) * \
                np.exp(-1j * mp * psi) * \
                complex(Rotation.D(S(int(2*j))/2, S(int(2*mp))/2,
                                   S(int(2*m_prime))/2, phi, -theta, -phi).doit())

        return result

    @staticmethod
    def cutoff_function(r, rc):

        if r > rc:
            return 0.

        else:
            return 0.5 * (np.cos(np.pi * r / rc) + 1.)


if __name__ == "__main__":
    # ---------------------- Options ------------------------
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

    f = Bispectrum(test)

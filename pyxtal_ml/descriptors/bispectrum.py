from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.periodic_table import Element, Specie
import numpy as np
from angular_momentum import CG, wigner_D
from optparse import OptionParser
import numba

@numba.njit(numba.f8(numba.f8, numba.f8),
            cache=True, nogil=True, fastmath=True)
def cosine_cutoff(r, rc):

    if r > rc:
        return 0.

    else:
        return 0.5 * (np.cos(np.pi * r / rc) + 1.)


@numba.njit(numba.void(numba.i8, numba.f8[:,:,:,:,:]),
            cache=True, nogil=True, fastmath=True)
def populate_cg_array(j_max, in_arr):
    '''
    Populate a 5-D array with the Clebsch-Gordon coupling coefficients
    given a j max value, this is intended for specific use with the bispectrum

    The intent is to use this at the highest level of execution to minimize computation

    We map the clebsch gordon arguments to array indeces using the following relations

    CG(J1, M1, J2, M2, J, M)

    Jn -> Jn/2

    Mn -> Mn - Jn/2


    args:
        j_max: specify the degree of bispectrum calculation

        in_arr: supplied array to insert CG coefficients
    '''
    # initiate a 5-d array with size 2*j_max+1
    twojmax = 2*j_max
    size = twojmax + 1
    cgs = in_arr
    # index j1 and j2 from 0 to 2jmax
    for j1 in range(size):
        for j2 in range(size):
            # there exists a symmetry for j given by this array
            js = np.arange(np.abs(j1-j2), min(twojmax, j1+j2) + 1, 2)
            for j in js:

                # index the sum from 0 to j1
                for m1 in range(j1+1):
                    aa2 = 2 * m1 - j1
                    #index the sym from 0 to j2
                    for m2 in range(j2+1):
                        bb2 = 2*m2-j2

                        # aa2 and bb2 ensure that m1 and m2 = m in the CG calc
                        m = (aa2+bb2+j)/2

                        '''
                            restrict z-angular momentum to be a positive value
                            not larger than total angular momentum
                        '''
                        if m < 0 or m > j:
                            continue

                        # convert array arguments to CG args
                        J1 = j1/2
                        J2 = j2/2
                        J = j/2

                        M1 = m1 - J1
                        M2 = m2 - J2
                        M = m - J
                        # add CG coef to cgs array
                        cgs[j1,j2,j,m1,m2] = CG(J1, M1, J2, M2, J, M)


@numba.njit(numba.c16(numba.i8, numba.i8, numba.i8,
                      numba.f8, numba.f8, numba.f8),
            cache=True, fastmath=True, nogil=True)
def U(j, m, m_prime, psi, theta, phi):
    '''
    Computes the 4-D hyperspherical harmonic given the three angular coordinates
    and indeces
    '''
    j = j/2
    m = m - j
    m_prime = m_prime - j
    sph_harm = 0. + 0.j
    mvals = np.arange(-j, j+1, 1)
    for mp in mvals:
        sph_harm += wigner_D(j, m, mp, phi, theta, -phi) * \
                np.exp(-1j * mp * psi) * \
                wigner_D(j, mp, m_prime, phi, -theta, -phi)

    return sph_harm.conjugate()


@numba.njit(numba.c16(numba.i8, numba.i8, numba.i8,
                      numba.f8[:,:], numba.f8[:], numba.f8[:]),
            cache=True, fastmath=True, nogil=True)
def compute_C(j, mp, m, hypersphere_coords, rbf_vals, cutoff_vals):
    '''
    Computes the inner product of the 4-D hyperspherical harmonics
    radial basis function values and cutoff function values

    Args:
        j, mp, m:  indeces for 4-D hyperspherical harmonics
    '''
    dot = 0
    for i in range(len(hypersphere_coords)):
        psi = hypersphere_coords[i,0]
        theta = hypersphere_coords[i,1]
        phi = hypersphere_coords[i,2]
        harmonic = U(j, m, mp, psi, theta, phi)
        dot += harmonic*rbf_vals[i]*cutoff_vals[i]

    return dot


@numba.njit(numba.void(numba.i8, numba.c16[:,:,:], numba.f8[:,:],
                       numba.f8[:], numba.f8[:]),
            cache=True, fastmath=True, nogil=True)
def populate_C_array(jmax, in_arr, hypersphere_coords, rbf_vals, cutoff_vals):
    '''
    Populates the array of the inner products from compute_C

    args:
        jmax: degree of bispectrum calculation
        in_arr: reference to array to populate
        hypersphere_coords: 2-D array of psis, thetas, phis from 4-d spherical coordinate system
        rbf_vals: 1-D array of radial basis function values
        cutoff_vals: 1-D array of cutoff function values

    note that these arrays should be the same length
    '''

    twojmax = 2*jmax
    size = twojmax + 1
    cs = in_arr

    for j in range(size):
        ms = np.arange(0, j+1, 1)
        for ma in ms:
            for mb in ms:
                cs[j, mb, ma] = compute_C(j, mb, ma, hypersphere_coords, rbf_vals, cutoff_vals)




@numba.njit(numba.void(numba.i8, numba.f8[:,:,:,:,:],
                       numba.c16[:,:,:], numba.c16[:,:,:,:,:]),
            cache=True, fastmath=True, nogil=True)
def compute_z_array(jmax, cgs, cs, in_arr):
    '''
    Precomputes the last two sums in the bispectrum

    args:
        jmax: degree of bispectrum
        cgs:  5-D array of clebsch gordon coefficients
        cs: 3D array of the inner products of the 4-D hyperspherical
            harmonics, radial basis function, and cutoff function
        in_arr: 5-D array for sums
'''


    twojmax = 2*jmax
    size = twojmax + 1
    zs = in_arr
    cgs = cgs
    cs = cs

    for j1 in range(size):
        for j2 in range(j1+1):
            js = np.arange(j1-j2, min(twojmax, j1+j2) + 1, 2)
            for j in js:
                mbs = np.arange(0, j/2 + 1, 1)
                for mb in mbs:
                    for ma in range(j+1):
                        ma1s = np.arange(max(0, (2*ma-j-j2+j1)/2),
                                         min(j1, (2*ma-j+j2+j1)/2) + 1,
                                         1)

                        for ma1 in ma1s:
                            sumb1 = 0. + 0.j
                            ma2 = (2 * ma- j - (2 * ma1 - j1) + j2) / 2

                            mb1s = np.arange(max(0, (2*mb-j-j2+j1)/2),
                                             min(j1, (2*mb-j+j2+j1)/2) + 1,
                                             1)

                            for mb1 in mb1s:
                                mb2 = (2 * mb - j - (2* mb1 - j1) + j2) / 2
                                sumb1 += cgs[int(j1),int(mb1),int(j2),int(mb2),int(j)] * cs[int(j1),int(ma1),int(mb1)] * cs[int(j2),int(ma2),int(mb2)]

                            zs[int(j1),int(j2),int(j),int(ma),int(mb)] += sumb1*cgs[int(j1),int(ma1),int(j2),int(ma2),int(j)]


@numba.njit(numba.void(numba.i8, numba.c16[:,:,:],
                       numba.c16[:,:,:,:,:], numba.c16[:,:,:]),
            cache=True, nogil=True, fastmath=True)
def compute_bispectrum(jmax, cs, zs, in_arr):
    '''
    Computes the bispectrum

    args:
        jmax: degree of bispectrum
        cs: 3D array of precomued inner products of
            hyperspherical harmonics, radial basis function,
            and cutoff function
        zs: 5-D array of pre computed sums (see SNAP)
    '''

    twojmax = 2*jmax
    size = twojmax + 1
    bis = in_arr

    for j1 in range(size):
        for j2 in range(size):
            js = np.arange(np.abs(j1-j2), min(twojmax, j1+j2) + 1, 2)
            for j in js:
                mbs = np.arange(0, j/2 + 1, 1)
                for mb in mbs:
                    for ma in range(j+1):
                        c = cs[int(j),int(ma),int(mb)]
                        bis[int(j1),int(j2),int(j)] += 2*c.conjugate()*zs[int(j1),int(j2),int(j),int(ma),int(mb)]


class Bispectrum(object):

    def __init__(self, crystal, j_max=5, cutoff_radius=6.5, symmetrize=True, CG_coefs=None):

        # populate private attributes
        self._j_max = j_max

        # symmetrize structure option
        if symmetrize:
            finder = SpacegroupAnalyzer(crystal, symprec=0.06,
                                        angle_tolerance=5)
            crystal = finder.get_conventional_standard_structure()

        '''
        Here we find all of the neighbors up to the specified cutoff radius,
        then we transform those x, y, and z coordinates to a 4-D hyperspherical
        coordinate system: r, psi, theta, phi.  Where r is the ordinary position
        vector magnitude, psi, theta and phi are angular coordinates defined in the
        code below.

        The code below populates arrays for the hypersphere angular coordinates,
        the radial cutoff function value, and the atomic numbers at each site, and stores
        them in an attribute.
        '''
        # neighbor calculation
        neighbors = crystal.get_all_neighbors(cutoff_radius)
        hypersphere_coords = []  # psi, theta, phi angles
        AN = []  # atomic numbers of sites
        cutoff_vals = [] # cutoff function values (radial component)
        for index, site in enumerate(crystal):
            '''
            first dimension corresponds to site,
            second dimension corresponds to neighbor
            third dimension corresponds to coordinates / values
            '''
            hypersphere_coords.append([])
            AN.append([])
            cutoff_vals.append([])

            for neighbor in neighbors[index]:
                '''
                Each neighbor in neighbors is a tuple of length 2

                The item at the zero index in the tuple is the pymatgen
                periodic site object

                The item at the one index in the tuple is the euclidian
                distance from the center site to the neighbor site
                '''

                # cartesian coordinates
                x, y, z = neighbor[0].coords - site.coords

                # euclidean vector norm
                r = neighbor[1]

                # do not populate the attribute for small r
                if r < 10.**(-10):
                    continue

                psi = np.arcsin(r / cutoff_radius)

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

                # [index] corresponds to a periodic site
                hypersphere_coords[index].append([psi, theta, phi])
                AN[index].append([neighbor[0].specie.number])

                # eventually we may want to consider different symmetry functions
                cutoff_vals[index].append([cosine_cutoff(r, cutoff_radius)])

        # populate the attributes
        self.hypersphere_coords = np.array(hypersphere_coords)
        self.atomic_numbers = np.array(AN)
        self.cutoff_vals = np.array(cutoff_vals)
        self.CGs = CG_coefs

    def _calculate_B(self, j1, j2, j, site, site_neighbors):
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
        # itiate the variable
        B = 0.
        # compute the set of free integer parameters
        mvals = self._m_values(j)
        # iterate over two free integer sets
        for m in mvals:
            for m_prime in mvals:
                # see calculate c
                c = self._calculate_c(j, m_prime, m, site, site_neighbors)
                # assign new integer parameters, related to m and m prime
                m1bound = min(j1, m + j2)
                m_prime1_bound = min(j1, m_prime + j2)
                m1 = max(-j1, m-j2)
                '''
                Iterate over a set of half integer values depending on the free
                integer parameters m and m_prime
                '''
                while m1 < (m1bound + 0.5):
                    m_prime1 = max(-j1, m_prime - j2)
                    while m_prime1 < (m_prime1_bound + 0.5):
                        c1 = self._calculate_c(j1, m_prime1, m1, site,
                                               site_neighbors)
                        c2 = self._calculate_c(j2, m_prime-m_prime1, m-m1, site,
                                               site_neighbors)

                        B += float(CG(j1, m1, j2, m-m1, j, m)) * \
                            float(CG(j1, m_prime1, j2, m_prime-m_prime1, j, m_prime)) * \
                            np.conjugate(c) * c1 * c2
                        m_prime1 += 1
                    m1 += 1

        return B



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

    #test = Structure.from_file(options.structure)

    #f = Bispectrum(test, j_max=1, cutoff_radius=6.5)
    #print(f.bispectrum)
    jmax = 1
    in_arr = np.zeros([2*jmax+1]*5)
    x = populate_cg_array(jmax, in_arr)
    print(in_arr)

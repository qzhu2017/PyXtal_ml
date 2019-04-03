from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.periodic_table import Element, Specie
import numpy as np
from angular_momentum import CG, wigner_D
from optparse import OptionParser
import numba

def cosine_cutoff(r, rc):
    '''
    Only cutoff function implemented this far
    '''
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

    We map the inner product arguments to array indeces using the following relations

    J -> J/2

    Mn -> Mn - J/2

    '''

    #array size
    twojmax = 2*jmax
    size = twojmax + 1
    #reference to input array
    cs = in_arr

    for j in range(size):
        ms = np.arange(0, j+1, 1)
        for ma in ms:
            for mb in ms:
                cs[j, mb, ma] = compute_C(j, mb, ma, hypersphere_coords, rbf_vals, cutoff_vals)




@numba.njit(numba.void(numba.i8, numba.f8[:,:,:,:,:],
                       numba.c16[:,:,:], numba.c16[:,:,:,:,:]),
            cache=True, fastmath=True, nogil=True)
def populate_z_array(jmax, cgs, cs, in_arr):
    '''
    Precomputes the last two sums in the bispectrum

    args:
        jmax: degree of bispectrum
        cgs:  5-D array of clebsch gordon coefficients
        cs: 3D array of the inner products of the 4-D hyperspherical
            harmonics, radial basis function, and cutoff function
        in_arr: 5-D array for sums

    We map the sum arguments to array indeces using the following relations

    Z(J1, M1, J2, M2, J, M)

    Jn -> Jn/2

    Mn -> Mn - Jn/2

'''


    twojmax = 2*jmax
    size = twojmax + 1
    zs = in_arr

    for j1 in range(size):
        for j2 in range(j1+1):
            js = np.arange(j1-j2, min(twojmax, j1+j2) + 1, 2)
            for j in js:
                mbs = np.arange(0, j/2 + 1, 1)
                mb = 0
                while 2*mb <= j:
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
                                sumb1 += cgs[int(j1),int(j2),int(j),int(mb1),int(mb2)] * cs[int(j1),int(ma1),int(mb1)] * cs[int(j2),int(ma2),int(mb2)]


                            zs[int(j1),int(j2),int(j),int(ma),int(mb)] += sumb1*cgs[int(j1),int(j2),int(j),int(ma1),int(ma2)]
                    mb += 1


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

    indices = [[0,0,0], [1,0,0], [1,1,2], [2,0,2], [2,2,2]]
    twojmax = 2*jmax
    size = twojmax + 1
    bis = in_arr

    for j1 in range(size):
        for j2 in range(j1+1):
            js = np.arange(np.abs(j1-j2), min(twojmax, j1+j2) + 1, 2)
            for j in js:
                if j1 > j:
                    continue
                mbs = np.arange(0, j/2 + 1, 1)
                mb = 0
                while 2*mb <= j:
                    for ma in range(j+1):
                        c = cs[int(j),int(mb),int(ma)]
                        z = zs[int(j1),int(j2),int(j),int(mb),int(ma)]
                        #if [int(j1), int(j2), int(j)] in indices:
                            #print('c = ',c, 'z = ', z)
                        bis[int(j1),int(j2),int(j)] += c.conjugate()*z
                    mb += 1

                if int(j)%2 == 0:
                    mb = int(j/2)
                    for ma in range(mb+1):
                        bis[int(j1),int(j2),int(j)] += cs[int(j),int(ma),int(mb)] *\
                                zs[int(j1),int(j2),int(j),int(ma),int(mb)]

                        if ma == mb:
                            bis[int(j1),int(j2),int(j)] *= 0.5

                if j != 0 and j != 1:
                    bis[int(j1),int(j2),int(j)] *= 2.0
    #print('\n\n')

class Bispectrum(object):

    def __init__(self, crystal, j_max=5, cutoff_radius=6.5, symmetrize=True, CG_coefs=None):

        # populate private attributes
        self._jmax = j_max
        self._size = 2*jmax+1

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
        self._neighbors = crystal.get_all_neighbors(cutoff_radius)
        max_size = 0
        for neighbors in self._neighbors:
            if max_size < len(neighbors):
                max_size = len(neighbors)

        hypersphere_coords = np.zeros([len(self._neighbors), max_size, 3])  # psi, theta, phi angles
        AN = np.zeros([len(self._neighbors), max_size])  # atomic numbers of sites
        cutoff_vals = np.zeros([len(self._neighbors), max_size]) # cutoff function values (radial component)
        for index, site in enumerate(crystal):

            for i, neighbor in enumerate(self._neighbors[index]):
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
                hypersphere_coords[index, i, :] = [psi, theta, phi]
                AN[index, i] = neighbor[0].specie.number

                # eventually we may want to consider different symmetry functions
                cutoff_vals[index, i] = cosine_cutoff(r, cutoff_radius)

        # populate the attributes
        self._hypersphere_coords = hypersphere_coords
        self._rbf_vals = AN
        self._cutoff_vals = cutoff_vals
        self._CGs = CG_coefs

    def precompute_arrays(self):
        neigh_size = len(self._neighbors)
        self._cs = np.zeros([neigh_size] + [self._size]*3, dtype=np.complex128)
        self._zs = np.zeros([neigh_size] + [self._size]*5, dtype=np.complex128)
        self._bis = np.zeros([neigh_size] + [self._size]*3, dtype=np.complex128)
        for i in range(neigh_size):
            populate_C_array(self._jmax, self._cs[i], self._hypersphere_coords[i],
                                np.array(self._rbf_vals[i]), np.array(self._cutoff_vals[i]))
            populate_z_array(self._jmax, self._CGs, self._cs[i], self._zs[i])
            compute_bispectrum(self._jmax, self._cs[i], self._zs[i], self._bis[i])

    def get_descr(self):
        self.precompute_arrays()

        for i, bis in enumerate(self._bis):
            if i == 0:
                bispectrum = np.ndarray.flatten(bis[np.nonzero(bis)])

            else:
                bispectrum = np.vstack([bispectrum, np.ndarray.flatten(bis[np.nonzero(bis)])])

        bispectrum = np.array(bispectrum, dtype=np.float64)
        return bispectrum



if __name__ == "__main__":
    # ---------------------- Options ------------------------
    parser = OptionParser()
    parser.add_option("-c", "--crystal", dest="structure", default='',
                      help="crystal from file, cif or poscar, REQUIRED",
                      metavar="crystal")


    parser.add_option("-r", "--rcut", dest="rcut", default=6.5, type=float,
                      help="cutoff for neighbor calcs, default: 2.0"
                     )

    parser.add_option("-j", "--jmax", dest="jmax", default=1, type=int,
                      help="jmax, default: 3"
                     )

    (options, args) = parser.parse_args()

    if options.structure.find('cif') > 0:
        fileformat = 'cif'
    else:
        fileformat = 'poscar'

    test = Structure.from_file(options.structure)
    jmax = options.jmax
    rcut = options.rcut

    in_arr = np.zeros([2*jmax+1]*5)
    populate_cg_array(jmax, in_arr)

    import time
    start = time.time()
    f = Bispectrum(test, j_max=jmax, cutoff_radius=rcut, CG_coefs=in_arr)
    bis = f.get_descr()
    end = time.time()

    print(bis)
    print(np.nonzero(f._bis[0]))
    print(f._bis[0,1,0,1])
    print('Computing the bispectrum of ', options.structure,
          'with jmax = ', jmax, 'with pre computed clebsch gordon coefficients takes: ',
          end-start, 'seconds')

import warnings
warnings.filterwarnings("ignore")
import ctypes
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.periodic_table import Element, Specie
from pyxtal_ml.descriptors.stats import descriptor_stats
import numpy as np
from optparse import OptionParser
import os

try: 
    lib = ctypes.CDLL(os.environ['lib_bispectrum'])
    lib.Bis.argtypes = [ctypes.c_int, ctypes.c_double, ctypes.c_int]
except:
    print("you must compile the lib_bispectrum.so and export the path to your .bashrc file")

def init_B(j_max):
    count = 0
    for i in range(2*j_max + 1):
        for j in range(min(i, j_max) + 1):
            count += 1

    return np.zeros(count)

def site_bispectrum(S, N, G, j_max, R):
    B = init_B(j_max)
    lib.Bis(j_max, R, len(G),
            S.ctypes.data_as(ctypes.c_void_p),
            N.ctypes.data_as(ctypes.c_void_p),
            G.ctypes.data_as(ctypes.c_void_p),
            B.ctypes.data_as(ctypes.c_void_p)
           )
    return B

class C_Bispectrum(object):

    def __init__(self, crystal, j_max=1, cutoff_radius=6.5):


        bispectrum = []
        neighbors = crystal.get_all_neighbors(cutoff_radius)

        for i, site in enumerate(crystal):
            S = site.coords
            N = []
            G = []

            for neighbor in neighbors[i]:
                N.append(neighbor[0].coords)
                G.append(neighbor[0].specie.number)


            N = np.reshape(np.array(N), (len(N)*3, 1))
            G = np.array(G).astype(float)
            B = site_bispectrum(S, N, G, j_max, cutoff_radius)

            bispectrum += [B]

        bispectrum = np.array(bispectrum)

        self.bispectrum = descriptor_stats(bispectrum, axis=0).get_stats()


if __name__ == "__main__":
    # ---------------------------Options----------------------------
    parser = OptionParser()
    parser.add_option("-c", "--crystal", dest="structure",
                      help="crystal from file, cif or poscar, REQUIRED"
                     )

    parser.add_option("-r", "--rcut", dest="rcut", default=6.5, type=float,
                      help="cutoff for neighbor calcs, default: 2.0"
                     )

    parser.add_option("-j", "--jmax", dest="jmax", default=1, type=int,
                      help="jmax, default: 3"
                     )

    parser.add_option("-s", "--symmetrize", dest="symmetrize", action='store_true',
                      help="symmetrize the structure"
                     )

    (options, args) = parser.parse_args()


    if options.structure.find('cif') > 0:
        fileformat = 'cif'

    else:
        fileformat = 'poscar'

    crystal = Structure.from_file(options.structure)

    if options.symmetrize:
        finder = SpacegroupAnalyzer(crystal, symprec=0.06,
                                    angle_tolerance=5)

        crystal = finder.get_conventional_standard_structure()

    j_max = options.jmax
    Rc = options.rcut

    bispectrum = C_Bispectrum(crystal, j_max, Rc)

    print(bispectrum.bispectrum, np.shape(bispectrum.bispectrum))

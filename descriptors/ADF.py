from pymatgen.core.periodic_table import Element
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
from optparse import OptionParser


def get_metals():
    metals = []
    for m in dir(Element)[:102]:
        ele = Element[m]
        if ele.is_transition_metal or ele.is_post_transition_metal \
            or ele.is_alkali or ele.is_alkaline:
            metals.append(m)
    return metals

def get_radii(ele):
    if ele.value in get_metals():
        return ele.metallic_radius
    else:
        return ele.atomic_radius


class ADF(object):
    """a class of crystal structure.
    Args:
        crystal: crystal class from pymatgen
        symmetrize: symmetrize the structure before computation
        A_bin: length of each bin when computing the ADF
    Attributes:
        crystal
        A_bin
        ADF
        plot_ADF()
    """

    def __init__(self, crystal, symmetrize=True, bin_width=1.0):
        """Return a ADF object with the proper info"""
        self.width = bin_width
        if symmetrize:
            finder = SpacegroupAnalyzer(crystal, symprec=0.06,
                                        angle_tolerance=5)
            crystal = finder.get_conventional_standard_structure()

        self.compute_angles(crystal)
        self.ADF, self.bins = np.histogram(self.angles, bins=np.arange(self.width, 181, self.width), density=True)

    def compute_angles(self, struc, R_max=4.0):
        """
        Computes the angle distribution function of a given crystal.
        Args:
        self: ADF
        crystal: crystal in pymatgen struc class
        R_max: the cutoff values for computing the neigbors.
        Returns: None
        """
        neighbors = struc.get_all_neighbors(r=R_max, include_index=True, include_image=True)
        angles = []
        for i, site in enumerate(struc.sites):
            rad1 = get_radii(site.specie) #.atomic_radius
            newlist = []
            #dist = []
            #for table in neighbors[i]:
            #    dist.append(table[1])
            #cutoff = min(dist) + 0.5
            # todo: change the cutoff distance by playing with the sets of atomic_radius/metallic_radius       
            for table in neighbors[i]:
                rad2 = get_radii(table[0].specie)
                #print(rad1, rad2)
                cutoff = 1.2*(rad1+rad2)
                if table[1] < cutoff: 
                    newlist.append(table)

        self.angles = np.hstack((angles, self.angles_from_atom(newlist, site))) 
        return
    
    @staticmethod
    def single_angle(p1, p2, p3):
        v1, v2 = p1-p2, p3-p2
        angle = np.dot(v1, v2)/np.linalg.norm(v1)/np.linalg.norm(v2)
    
        return np.degrees(np.arccos(angle))
    
    def angles_from_atom(self, table, site):
        angles = []
        p2 = site.coords
        if len(table) > 1:
            for i in range(len(table)-1):
                p1 = table[i][0].coords
                for j in range(i+1, len(table)):
                    p3 = table[j][0].coords
                    angles = np.hstack((angles, self.single_angle(p1, p2, p3)))
                    #print(table[i][0].specie, site.specie, table[j][0].specie, self.single_angle(p1, p2, p3) )
        return angles

    def plot_ADF(self, filename=None):
        plt.hist(self.angles, bins=np.arange(self.width, 181, self.width), density=True)
        plt.grid()
        plt.xlabel("Angle (degree)", fontsize=16)
        plt.ylabel("ADF", fontsize=16)
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename)
            plt.close()


if __name__ == "__main__":
    # -------------------------------- Options -------------------------
    parser = OptionParser()
    parser.add_option("-c", "--crystal", dest="structure", default='',
                      help="crystal from file, cif or poscar, REQUIRED",
                      metavar="crystal")
    parser.add_option("-d", "--delta", dest="delta", default=1.0,
                      type='float', help="step length, default: 0.08",
                      metavar="R_bin")
    parser.add_option("-o", "--output", dest="mstyle",
                      default='bmh',
                      help="matplotlib style, fivethirtyeight, bmh, grayscale, dark_background, ggplot",
                      metavar="mstyle")

    (options, args) = parser.parse_args()
    if options.structure.find('cif') > 0:
        fileformat = 'cif'
    else:
        fileformat = 'poscar'

    plt.style.use(options.mstyle)
    test = Structure.from_file(options.structure)
    adf = ADF(test)
    print('-----ADF value-----')
    print(adf.ADF)
    adf.plot_ADF()

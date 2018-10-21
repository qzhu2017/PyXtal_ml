from pymatgen.core.periodic_table import Element
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from scipy.spatial.distance import cdist
import numpy as np
import math
import matplotlib.pyplot as plt
from optparse import OptionParser

def angle_from_ijk(p1, p2, p3):
    """
    compute angle from three points
    """
    v1, v2 = p1-p2, p3-p2
    angle = np.degrees(np.arccos(np.dot(v1, v2)/np.linalg.norm(v1)/np.linalg.norm(v2)))
    #QZ: sometimes, it returns nan due to numerical error (with cos value slightly > 1))
    if abs(angle) < 5e-1 or np.isnan(angle): 
        angle = 180
    #if np.isnan(angle):  
    #    print('Warning! return NaN value --')
    #    print('INPUT:', p1, p2, p3)
    #    print('cosine values:', np.dot(v1, v2)/np.linalg.norm(v1)/np.linalg.norm(v2))

    return angle

def angle_from_ijkl(p1, p2, p3, p4):
    """
    compute dihedral angle from four points, we consider only -90 to 90 for simplicity
    """
    if angle_from_ijk(p1, p2, p3) == 180 or angle_from_ijk(p2, p3, p4) == 180:
        return 0.0
    else:
        v1, v2, v3 = p3-p4, p2-p3, p1-p2
        v23 = np.cross(v2, v3)
        v12 = np.cross(v1, v2)
        angle = np.degrees(math.atan(np.linalg.norm(v2) * np.dot(v1, v23)/np.dot(v12, v23)))
        #if angle < 0: print(np.linalg.norm(v2) * np.dot(v1, v23)/np.dot(v12, v23), angle)
        if abs(angle+90.0) < 5e-1: angle = 90

        return angle
 
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
        if ele.atomic_radius is None: #['He', 'Ne', 'Ar', 'Kr', 'Xe']
            rad = {'He': 0.28, 
                   'Ne': 0.58, 
                   'Ar': 1.06, 
                   'Kr': 1.16, 
                   'Xe': 1.40,
                   'At': 1.50,
                   'Rn': 1.50,
                  }
            return rad[ele.value]
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

    def __init__(self, crystal, calc_dihedral=False, symmetrize=True, bin_width=2.0):
        """Return a ADF object with the proper info"""
        self.width = bin_width
        self.calc_dihedral = calc_dihedral
        if symmetrize:
            finder = SpacegroupAnalyzer(crystal, symprec=0.02, angle_tolerance=5)
            #crystal = finder.get_conventional_standard_structure()
            crystal = finder.get_primitive_standard_structure()
        if calc_dihedral is True and len(crystal.sites) == 1:  
            crystal.make_supercell([1, 1, 2])
        self.get_neighbor_table(crystal)
        self.compute_angles(crystal.cart_coords)
        if len(self.angles) == 0:
            ADF = np.zeros(int(180/self.width))
        else:
            ADF, bins = np.histogram(self.angles, bins=int(180/self.width), range=(0.5, 180.5))
            ADF = np.array(ADF/ADF.sum())

        self.ADF = ADF
        if self.calc_dihedral:
            if len(self.torsion_angles) == 0:
                DDF = np.zeros(int(180/self.width))
            else:
                DDF, bins = np.histogram(self.torsion_angles, bins=int(180/self.width), range=(-89.5, 90.5))
                DDF = np.array(DDF/DDF.sum())
            self.DDF = DDF

        self.merge()
        #print(crystal.formula, self.angles)

    def merge(self):
        self.all = self.ADF
        if self.calc_dihedral:
            self.all = np.hstack((self.all, self.DDF))

    def get_neighbor_table(self, struc, R_max=4.0):
        """
        Computes neighbor table for a given crystal.
        Args:
        crystal: crystal in pymatgen struc class
        R_max: the cutoff values for computing the neigbors.
        Returns: table, cartesian coordinate
        """
        
        neighbors = struc.get_all_neighbors(r=R_max, include_index=True, include_image=True)
        rad = []
        for site in struc.sites:
            rad.append(get_radii(site.specie))

        table_ids = []
        table_coords = []
        table_dists = []
        for i, site in enumerate(struc.sites):
            list_coor = []
            list_id = []
            list_dist = []
            for table in neighbors[i]:
                cutoff = 1.2*(rad[i]+rad[table[2]])
                if table[1] < cutoff: 
                    list_coor.append(table[0].coords)
                    list_id.append(table[2])
                    list_dist.append(table[1])
            table_coords.append(list_coor)
            table_ids.append(list_id)
            table_dists.append(list_dist)

        self.table_coords = table_coords
        self.table_ids = table_ids
        self.table_dists = table_dists
    
    def compute_angles(self, cart_coords):
        angles = []
        if self.calc_dihedral:
            d_angles = []
        table_coords = self.table_coords
        table_ids = self.table_ids
        table_dists = self.table_dists
        for p2, list_coor, ids, list_dist in zip(cart_coords, table_coords, table_ids, table_dists):
            for m1, p1 in enumerate(list_coor[:-1]): # p1-p2
                l_list1 = ids[m1]
                coor_list1 = table_coords[ids[m1]]
                dist_list1 = table_dists[ids[m1]]
                for m2, p3 in enumerate(list_coor[m1+1:]): # p1-p2-p3
                    angles.append(angle_from_ijk(p1, p2, p3))
                    if self.calc_dihedral:
                        l_list2 = ids[m2]
                        coor_list2 = table_coords[ids[m2]]
                        dist_list2 = table_dists[ids[m2]]
                        disp = p1 - cart_coords[ids[m1]]
                        #print(disp)
                        for p4, dist in zip(coor_list1, dist_list1): #p4-p1-p2-p3
                            #print('---------------------------',np.linalg.norm(p4+disp-p1))
                            if abs(np.linalg.norm(p4-p1)-dist)<1e-1 and sum(abs(p4-p2))>5e-1 and sum(abs(p4-p3))>5e-1:
                                d_angles.append(angle_from_ijkl(p4+disp, p1, p2, p3))
                        disp = p3 - cart_coords[ids[m1]]
                        #print(disp)
                        for p4, dist in zip(coor_list2, dist_list2): #p1-p2-p3-p4
                            #print('---------------------------',np.linalg.norm(p4+disp-p3))
                            if abs(np.linalg.norm(p4-p3)-dist)<1e-1 and sum(abs(p4-p2))>5e-1 and sum(abs(p4-p1))>5e-1:
                                #print(np.linalg.norm(p4-p3), angle_from_ijkl(p1, p2, p3, p4))
                                d_angles.append(angle_from_ijkl(p1, p2, p3, p4))
        self.angles = angles
        if self.calc_dihedral:
            self.torsion_angles = d_angles

   
    def plot_ADF(self, filename=None):
        plt.hist(aelf.angles, bins=int(180/self.width), range=(0.5, 180.5), density=True)
        plt.grid()
        plt.xlabel("Angle (degree)", fontsize=16)
        plt.ylabel("ADF", fontsize=16)
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename)
            plt.close()

    def plot_DDF(self, filename=None):
        plt.hist(self.torsion_angles, bins=int(180/self.width), range=(-89.5, 90.5), density=True)
        plt.grid()
        plt.xlabel("Angle (degree)", fontsize=16)
        plt.ylabel("DDF", fontsize=16)
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
    adf = ADF(crystal=test, calc_dihedral=True, symmetrize=False)
    #print(adf.angles)
    #print(adf.torsion_angles)
    print(adf.ADF)
    print(adf.all)
    print(adf.DDF)
    test.make_supercell([2, 2, 2])
    adf = ADF(crystal=test, calc_dihedral=True, symmetrize=False)
    print(adf.ADF)
    print(adf.DDF)
    test.make_supercell([2, 2, 2])
    adf = ADF(crystal=test, calc_dihedral=True, symmetrize=False)
    print(adf.ADF)
    print(adf.DDF)

    #adf.plot_DDF()

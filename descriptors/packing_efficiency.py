from pymatgen.analysis.local_env import VoronoiNN
import numpy as np
from optparse import OptionParser
from pymatgen.core.structure import Structure


class packing_efficiency(object):
    '''
    Computes the maximum packing efficiency of a given cell using voronoi
    polyhedra

    Args:
        crystal:  crystal class from pymatgen
    Attributes:
        packing_efficiency: the maximum packing_efficiency of the crystal
    '''

    def __init__(self, crystal):
        # call the Voronoi class
        voronoi = VoronoiNN()
        # compute the voronoi polyhedra for each atom in the structure
        polyhedra = [voronoi.get_voronoi_polyhedra(crystal, i)
                     for i, _ in enumerate(crystal)]
        # find the distance from the center of the atom to the closest voronoi
        # face for each atom in the structure
        maximum_radii = [min(face['face_dist']
                             for face in polyhedron.values())
                         for polyhedron in polyhedra]
        # calculate the volume of each atomic occupancy, sum those volumes up
        # and divide by the total volume of the structure
        self.eff = [4/3 * np.pi *
                    np.power(maximum_radii, 3).sum()
                    / crystal.volume]


if __name__ == "__main__":
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
    packing_eff = packing_efficiency(test)
    print(packing_eff.eff)

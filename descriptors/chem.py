from collections import defaultdict
from pymatgen.core.structure import Structure
import numpy as np
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from monty.json import MontyEncoder, MontyDecoder
from pymatgen.core.structure import Structure
import numpy as np
from monty.serialization import loadfn
from pymatgen.core.lattice import Lattice
from pymatgen.core import Composition

el_chem_json=str(os.path.join(os.path.dirname(__file__),'Elements.json'))


def get_descrp_arr(elm=''):
    arr = []
    try:
        f = open(el_chem_json, 'r')
        dat = json.load(f)
        f.close()

        d = dat[elm]
        arr = []
        for k, v in d.items():
            arr.append(v)
        arr = np.array(arr).astype(float)
    except:
        pass
    return arr


def get_chemonly(struc):
    comp = struc.composition
    el_dict = comp.get_el_amt_dict()
    arr = []
    for k, v in el_dict.items():
        des = get_descrp_arr(k)
        arr.append(des)
    mean_chem = np.mean(arr, axis=0)
    return mean_chem

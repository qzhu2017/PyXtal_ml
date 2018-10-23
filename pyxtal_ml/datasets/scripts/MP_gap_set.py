from pymatgen.core.periodic_table import Element
from pymatgen.core import Structure
import numpy as np
from pymatgen import MPRester
import json
import sys
import os

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def read_struc(entry):
    lattice = entry['structure'].lattice.matrix
    coords = entry['structure'].frac_coords
    name_array = []
    for i in entry['structure'].species:
        name_array.append(i.value)
    formula = entry['structure'].formula
    struc = {'formula': formula,
            'coords': coords,
            'lattice': lattice,
            'elements': name_array,
            'formation_energy': entry['formation_energy_per_atom'],
            'band_gap': entry['band_gap'],
            }
    return struc

def read_json(json_file):
    with open(json_file, "r") as f:
        content = json.load(f)
    entry = []
    E_form = []
    for dct in content:
        lattice = dct['lattice']
        coords = dct['coords']
        elements = dct['elements']
        E_form.append(dct['formation_energy'])
        entry.append(Structure(lattice, elements, coords))

    return entry, E_form

m = MPRester('ZkyR13aTi9z5hLbX')
entries = m.query(criteria = {#"elements": {"$in": ['Li']},
                              "icsd_ids.0": {'$exists': True},
                              "nsites":{"$lte": 20},
                              "band_gap": {"$gt": 0.1},
                             }, 
                  properties = ["formula", 
                                "structure",
                                "formation_energy_per_atom",
                                "band_gap",
                             ]
                 )

struc_info = []
for entry in entries:
    struc_info.append(read_struc(entry))
json_file = 'nonmetal_MP_' + str(len(entry)) + '.json'

with open(json_file, "w") as f:
    json.dump(struc_info, f, cls=NumpyEncoder, indent=1)

entry, E_form = read_json(json_file)
print(len(E_form))

from pymatgen.core.periodic_table import Element
import numpy as np
from pymatgen import MPRester
import json
import sys

def get_s_metal():
    metals = []
    for m in dir(Element)[:102]:
        ele = Element[m]
        if ele.is_alkali or ele.is_alkaline:
            metals.append(m)
    return metals


def get_p_metal():
    metals = []
    for m in dir(Element)[:102]:
        ele = Element[m]
        #if ele.is_metalloid or ele.is_post_transition_metal:
        if ele.is_post_transition_metal:
            metals.append(m)
    return metals


def get_d_metal():
    metals = []
    for m in dir(Element)[:102]:
        ele = Element[m]
        if ele.is_transition_metal or ele.is_actinoid or ele.is_lanthanoid:
            metals.append(m)
    metals.append('Zr')
    return metals


def get_non_metal():
    elements = []
    metals = get_s_metal() + get_p_metal() + get_d_metal()
    for m in dir(Element)[:102]:
        if m not in metals:
            elements.append(m)
    return elements 


s = get_s_metal()
p = get_p_metal()
d = get_d_metal()
nm = get_non_metal()

print(s+p)
print(d)
m = MPRester('ZkyR13aTi9z5hLbX')
entries = m.query(criteria = {"elements": {"$in": s+p+nm},
                              #"icsd_ids.0": {'$exists': True},
                              "nsites":{"$lte": 6},
                              "band_gap": 0.0,
                             }, 
                  properties = ["pretty_formula", 
                                "material_id",
                                "structure",
                                # "dos", dos tag fails to return the results
                                "spacegroup.symbol",
                                "elements",
                             ]
                 )

print(len(entries))
accepted = []
pool = d
print(nm+d)

for entry in entries:
    add = True
    for element in entry['elements']:
        if element in pool:
            add = False
            break

    if add:
        accepted.append(entry)
#        try:
#            id = entry['material_id']
#            entry['dos'] = m.get_dos_by_material_id(id).as_dict()
#            entry['structure'] = entry['structure'].as_dict()
#            accepted.append(entry)
#        except:
#            print('no dos info for ', id, entry['pretty_formula'])
#
#
print(len(accepted))
#
#for entry in accepted: print(entry["pretty_formula"])
#dumped = json.dumps(accepted, indent=2)
#with open("sp.json","w") as f:
#    f.write(dumped)

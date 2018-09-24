from pymatgen.core.periodic_table import Element
import numpy as np
from pymatgen import MPRester
import json
import sys
import os

def get_icsd():
    icsd_ids = []
    os.chdir('prb_data')
    for file in os.listdir("./"):
        if file.endswith(".dos"):
            icsd_ids.append(int(file[:-4]))
            #print(file[:-4])
    return icsd_ids


m = MPRester('ZkyR13aTi9z5hLbX')
entries = m.query(criteria = {#"elements": {"$in": s+p+nm},
                              "icsd_ids.0": {'$exists': True},
                              #"nsites":{"$lte": 6},
                              #"band_gap": {"$lt": 0.2},
                             }, 
                  properties = ["pretty_formula", 
                                "material_id",
                                #"structure",
                                # "dos", dos tag fails to return the results
                                "spacegroup.symbol",
                                "icsd_ids",
                                "nsites",
                                "elements",
                             ]
                 )

icsd_ids = get_icsd()
print(len(entries), len(icsd_ids))
accepted = []

for entry in entries:
    add = False
    for id in entry["icsd_ids"]:
        if id in icsd_ids:
            add = True
            break
    if add:
        accepted.append(entry)
print(len(accepted))

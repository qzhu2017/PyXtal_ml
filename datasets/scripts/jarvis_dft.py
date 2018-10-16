from monty.serialization import loadfn, MontyDecoder,MontyEncoder
import json

dat_3d=loadfn('jdft_3d-7-7-2018.json',cls=MontyDecoder)
struc_info = []
for dat in dat_3d:
    struc = dat['final_str']
    formation_energy = dat['form_enp']
    gap = dat['op_gap']
    lattice = struc.lattice.matrix
    coords  = struc.frac_coords
    name_array = []
    for i in struc.species:
        name_array.append(i.value)
    formula = struc.formula
    dic = {'formula': formula,
           'coords': coords,
           'lattice': lattice,
           'elements': name_array,
           'formation_energy': formation_energy, 
           'band_gap': gap,
           }
    struc_info.append(dic)

json_file = 'jarvis_' + str(len(dat_3d)) + '.json'
with open(json_file, "w") as f:
    json.dump(struc_info, f, cls=MontyEncoder, indent=1)


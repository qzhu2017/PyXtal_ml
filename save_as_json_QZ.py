# construct json

import os
import sys
import lzma
import numpy as np
import json
from aflow import *
from Descriptors.RDF import RDF
from pymatgen.core.composition import Composition
from pymatgen.core import Lattice
from pymatgen.core.periodic_table import Element
from pymatgen.core import Structure

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

# Functions
def save_xz(filename, URL):
    """
    1. Save .xz zipfile downloaded from an online database.
    2. Unzip the zipped files.
    
    Args:
        URL: provide a URL of the database to look for the zipfile.
        filename: provide the name of the file; filename should end with '.xz'.
    """
    URL(filename)
    zipfile = lzma.LZMAFile(filename).read()
    newfilepath = filename[:-3]
    fo = open(newfilepath+'.txt', 'wb').write(zipfile)
    os.remove(filename)
    
def get_DOS_fermi(filename, volume):
    """
    This function takes DOS file and return intensities near the fermi level.
    
    Args:
        filename: provide the DOS file; filename should end with '.txt'.
        
        volume: input the material entry to include volume in the DOS.
        
    Returns:
        DOS at fermi level
    """
    with open(filename, 'r') as fin:
        dataf = fin.read().splitlines(True)
        fin.close()
    with open(filename, 'w') as fout:
        E_Fermi = [float(i) for i in dataf[5].split()][3]
        fout.writelines(dataf[6:5006])
        fout.close()
    
    Volume = volume.volume_cell
    DOS = np.genfromtxt(filename, dtype = float)
    energy = DOS[:,0] - E_Fermi
    dos = DOS[:,1]/Volume                           # 1/(eV*A^3)
    combine = np.vstack((energy, dos))
    combine_abs = abs(combine[0,:])
    find_ele_at_fermi = np.where(combine_abs == min(combine_abs))
    ele_at_fermi = find_ele_at_fermi[0][0]
    
    return combine[1,ele_at_fermi-3:ele_at_fermi+4]

def get_s_metal():
    """
    get all metallic elements in group 1 & 2.
    
    Returns:
        an array of metallic elements in group 1 & 2.
    """
    metals = []
    for m in dir(Element)[:102]:
        ele = Element[m]
        if ele.is_alkali or ele.is_alkaline:
            metals.append(m)
    return metals

def get_p_metal():
    """
    get all metallic elements in group 13 to 17.
    
    Returns:
        an array of metallic elements in group 13 to 17.
    """
    metals = []
    for m in dir(Element)[:102]:
        ele = Element[m]
        if ele.is_post_transition_metal:
            metals.append(m)
    return metals

def get_d_metal():
    """
    get all transition-metal elements.
    
    Returns:
        an array of transition-metal elements.
    """
    metals = []
    for m in dir(Element)[:102]:
        ele = Element[m]
        if ele.is_transition_metal:
            metals.append(m)
    metals.append('Zr')
    return metals

def read_json(json_file):
    with open(json_file, "r") as f:
        content = json.load(f)
    entry = []
    E_form = []
    for dct in content:
        lattice = dct['lattice']
        coords = dct['coordinates']
        elements = dct['atom_array']
        E_form.append(dct['form_energy_cell'])
        entry.append(Structure(lattice, elements, coords))
    return entry, E_form
    
def material_properties(result, dos):
    """
    
    """
    atoms = []
    geometry = []
    for i, species in enumerate(result.species):
        for j in range(result.composition[i]):
            atoms.append(species)
    geo = result.geometry
    lat = Lattice.from_lengths_and_angles(geo[:3], geo[3:])

    mat_property = {'formula': result.compound,
                    'lattice': lat.matrix,
                    'coordinates': result.positions_fractional,
                    'atom_array': atoms,
                    'form_energy_cell': result.enthalpy_formation_cell,
                    'n_atoms': result.natoms,
                    'volume': result.volume_cell,
                    'space_group': result.spacegroup_relax,
                    'dos_fermi': dos}
    return mat_property #, print(atoms)

#materials_info.append(material_properties(results[54]))

####################################### Part a: Mining ###########################################
# Get materials from AFLOW database based on the given criteria: 
# sp metals with less than 7 different elements.

sp_system = get_s_metal() + get_p_metal()

results = search(batch_size = 100
                ).filter(K.Egap_type == 'metal'
                ).filter(K.nspecies < 7)

n = len(results) # number of avaiable data points

#d['dos'] = get_DOS_fermi(results[0].compound+'.txt', results[0])
#d['form_energy'] = results[0].enthalpy_formation_cell

X_sp_metals = []
Y_sp_metals = []
materials_info = []

for i, result in enumerate(results[51:100]):
    try:
        if result.catalog == 'ICSD\n':
            URL = result.files['DOSCAR.static.xz']
            #save_xz(result.compound+'.xz', URL)
    
            # Construct RDF with POSCAR
            crystal = Structure.from_str(result.files['CONTCAR.relax.vasp'](), fmt='poscar')
            
            # Get elements in the compound
            elements = result.species
            last_element = elements[-1]
            last_element = last_element[:-1]
            elements[-1] = last_element
            
            # Collecting for sp_metals compound
            j = 0
            for element in elements:
                if element in sp_system:
                    j += 1
            if j == len(elements):
                #X_sp_metals.append(RDF(crystal).RDF[1,:])
                #dos = get_DOS_fermi(result.compound+'.txt', result)
                #Y_sp_metals.append(dos)
                materials_info.append(material_properties(result, dos=1))
                
                print('progress: ', i+1, '/', n, '-------- material is stored')
            else:
                print('progress: ', i+1, '/', n, '-------- material is rejected')
            
        #os.remove(result.compound+'.txt')
            
    except:
        print('progress: ', i+1, '/', n, '-------- material does not fit the criteria')
        os.remove(result.compound+'.txt')
        pass

# Save as json for sp metals
with open('sp_metal_aflow_844.json', 'w') as f:
    json.dump(materials_info, f, cls=NumpyEncoder, indent=1)
    
entry, E_form = read_json('sp_metal_aflow_844.json')
print(entry, E_form)

results[0].lattice_variation_relax
